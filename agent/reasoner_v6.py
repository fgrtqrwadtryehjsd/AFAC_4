"""V6 推理引擎 — Query-Focused Evidence Extraction (QFEE)

核心思路（受 FinCARDS + Acon + Self-RAG 论文启发）：
- 不做全局文档压缩（压缩丢信息 = 准确率低）
- 每道题单独提取问题相关的证据段落（QFEE）
- 关键词粗筛 + BM25 双路召回 → 融合去重
- 直接用精选证据做 CoT 推理（无压缩损失）

流程：
1. 从问题+选项中提取关键词（query intent）
2. BM25 + 关键词精确匹配 → 候选 chunks（高召回）
3. 融合去重，取 top-K
4. 精选证据 → CoT 推理 → 答案
5. Self-Critique（多选题二次校验）
"""
import os
import json
import re
import time
from agent.config import RESULTS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.postprocessor import extract_answer_from_response


# ============ Prompt Engineering ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": """你是一位资深保险精算师和条款解读专家。
严格依据文档原文回答问题，不用常识推断。

推理规则：
- 身故保险金 ≠ 已交保费 ≠ 现金价值 ≠ 账户价值，必须区分
- 退保金额 = 现金价值 - 退保费用（注意退保费用随年度递减）
- 保险责任必须满足所有条件：等待期后、在保险期间内、非免责
- 计算题必须列公式和中间结果""",

    "regulatory": """你是一位金融监管合规专家。
严格依据法规原文判断，不用常识。

推理规则：
- "应当"/"必须"/"不得"= 强制性；"可以"= 授权性
- "经股东大会审议"≠"经特别决议通过"（后者需 2/3 表决权）
- 法规冲突：新法优于旧法，特别优于一般
- 合规判断必须逐条对照原文""",

    "financial_contracts": """你是一位金融合同分析专家。
严格依据合同条款原文判断。

推理规则：
- 主体评级 ≠ 债项评级
- 违约事件必须满足合同明确定义
- 偿付顺序按合同约定，不按常识""",

    "financial_reports": """你是一位财务报表分析专家。
严格依据年报披露的精确数字，不估算。

推理规则：
- 同比 = 去年同期；环比 = 上期
- "拟派发"(预案) ≠ "已派发"(实际)
- 经营/投资/筹资现金流严格区分
- 计算列出公式和中间值""",

    "research": """你是一位行业研报分析专家。
严格依据研报数据判断。

推理规则：
- "预计/预期" ≠ "实际"
- 研报数据优先于文字描述
- 图表数据最权威""",
}


# ============ Stage 1: 关键词提取 + 双路召回 ============

def extract_query_keywords(question: str, options: dict) -> list:
    """从问题和选项中提取关键词（FinCARDS Query Intent Mapping 思路）"""
    keywords = set()
    
    stopwords = {'的','了','在','是','和','与','或','及','等','对','为','从','中','不','有','这',
                 '那','也','都','就','而','但','如','其','以','所','可','将','被','让','给','比',
                 '到','对','于','和','或','以下','关于','上述','下列','哪些','哪个','是否','属于',
                 '以下哪些','下列哪些','描述','说法','结论','正确','准确','成立','判断'}
    
    for pattern in [r'[\u4e00-\u9fff]{2,6}', r'\d+\.?\d*%?', r'[A-Z]{2,}']:
        for m in re.finditer(pattern, question):
            w = m.group()
            if w not in stopwords and len(w) >= 2:
                keywords.add(w)
    
    for opt_text in options.values():
        for pattern in [r'[\u4e00-\u9fff]{2,6}', r'\d+\.?\d*%?', r'[A-Z]{2,}']:
            for m in re.finditer(pattern, opt_text):
                w = m.group()
                if w not in stopwords and len(w) >= 2:
                    keywords.add(w)
    
    return list(keywords)


def keyword_match_score(keywords: list, text: str) -> float:
    """关键词匹配评分"""
    score = 0.0
    for kw in keywords:
        count = text.count(kw)
        if count > 0:
            weight = max(1, len(kw) - 1) if len(kw) <= 3 else len(kw)
            score += count * weight
    return score


def rrf_fuse(bm25_results: list, kw_results: list, k: int = 60,
             bm25_weight: float = 1.0, kw_weight: float = 1.5) -> list:
    """RRF 融合 BM25 和关键词匹配结果
    
    关键词匹配权重更高（金融术语精确匹配比语义更可靠）
    """
    doc_scores = {}
    
    for rank, chunk in enumerate(bm25_results):
        key = chunk.get("text", "")[:100]
        rrf_score = bm25_weight / (k + rank + 1)
        if key not in doc_scores:
            doc_scores[key] = {"score": 0, "chunk": chunk}
        doc_scores[key]["score"] += rrf_score
    
    for rank, chunk in enumerate(kw_results):
        key = chunk.get("text", "")[:100]
        rrf_score = kw_weight / (k + rank + 1)
        if key not in doc_scores:
            doc_scores[key] = {"score": 0, "chunk": chunk}
        doc_scores[key]["score"] += rrf_score
    
    sorted_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["chunk"] for item in sorted_results]


# ============ Stage 2: CoT 推理 ============

COT_REASONING_PROMPT = """## 任务
基于以下文档证据，准确回答问题。

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 答题要求
1. 严格依据文档证据判断，不用常识和外部知识
2. {format_specific}

请按思维链逐步推理：

**步骤1：定位关键信息**
从文档证据中，找到与问题直接相关的关键事实、条款、数字。

**步骤2：逐选项分析**
对每个选项，判断文档是否有明确支持：
- 选项A：[有明确支持/无明确支持] — 引用：[原文关键语句]
- 选项B：[有明确支持/无明确支持] — 引用：[原文关键语句]
- 选项C：[有明确支持/无明确支持] — 引用：[原文关键语句]
- 选项D：[有明确支持/无明确支持] — 引用：[原文关键语句]

**步骤3：综合判断**
{judgment_hint}

最终答案：{answer_hint}"""


SELF_CRITIQUE_PROMPT = """你对一道多选题给出的答案是：{answer}

请进行严格二次校验：
- 对已选的每个选项，再次确认文档中是否有**明确原文**支持。如果只是间接推断或模糊相关，必须删除。
- 对未选的选项，不要添加（宁可漏选不可多选）。

你的原始推理过程：
{raw_response}

确认后的最终答案（只能从原始答案中删除选项，不能添加，多个大写字母按字母序）："""


FORMAT_SPECIFIC = {
    "mcq": "单选题：只有一个正确选项，排除法最有效",
    "tf": "判断题：A=正确，B=错误。找反例比找正例更有效",
    "multi": "多选题：只有文档明确支持的才选。找不到明确证据则必须排除。宁可少选不可多选。",
}

JUDGMENT_HINTS = {
    "mcq": "找出唯一有明确文档支持的选项，其余排除",
    "tf": "如果在文档中找不到支持原命题的证据，或者找到反例，则选B",
    "multi": "只保留有明确文档原文支持的选项，删除任何只有间接推理的选项",
}

ANSWER_HINTS = {
    "mcq": "一个大写字母(A/B/C/D)",
    "tf": "A或B",
    "multi": "多个大写字母按字母序排列(如ABC)",
}


def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options.keys()))


# ============ 推理 Agent V6 ============

class ReasoningAgentV6:
    """V6 推理 Agent — Query-Focused Evidence Extraction (QFEE)
    
    流程：
    1. 从问题+选项提取关键词
    2. BM25 + 关键词匹配 → RRF融合（高召回+高精度）
    3. 精选证据 → CoT 推理 → 答案
    4. Self-Critique（多选题二次校验）
    """
    
    def __init__(self, qwen: QwenClient, doc_index: DocumentIndex, 
                 token_budget: int = 5_000_000):
        self.qwen = qwen
        self.doc_index = doc_index
        self.token_budget = token_budget
        self.cot_trails = []
    
    def answer_question(self, question: dict) -> dict:
        """回答单道题"""
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])
        
        # Step 1: 提取关键词
        keywords = extract_query_keywords(q_text, options)
        
        # Step 2: 双路召回 + RRF 融合
        bm25_results = self.doc_index.search_bm25(q_text, top_k=30, doc_ids=doc_ids)
        
        all_doc_chunks = self.doc_index.get_chunks_by_doc_ids(doc_ids)
        kw_scored = []
        for chunk in all_doc_chunks:
            text = chunk.get("text", "")
            score = keyword_match_score(keywords, text)
            if score > 0:
                kw_scored.append((score, chunk))
        kw_scored.sort(key=lambda x: -x[0])
        kw_results = [chunk for _, chunk in kw_scored[:50]]
        
        # RRF 融合
        merged = rrf_fuse(bm25_results, kw_results, bm25_weight=1.0, kw_weight=1.5)
        
        # 取 top-20 作为证据
        evidence_chunks = merged[:20]
        
        # Step 3: 构建证据文本
        evidence_text = ""
        for chunk in evidence_chunks:
            doc_id = chunk.get("doc_id", "")
            text = chunk.get("text", "")
            evidence_text += f"\n--- 来自文档 {doc_id} ---\n{text}\n"
        
        # 控制证据总长度（不超过 15000 字符，留足推理空间）
        max_evidence_chars = 15000
        if len(evidence_text) > max_evidence_chars:
            # 保留前面的（更相关的）证据
            evidence_text = evidence_text[:max_evidence_chars] + "\n[...更多证据已省略...]"
        
        # Step 4: CoT 推理
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")
        reasoning_prompt = COT_REASONING_PROMPT.format(
            evidence=evidence_text,
            question=q_text,
            options=format_options(options),
            format_specific=FORMAT_SPECIFIC.get(answer_format, ""),
            judgment_hint=JUDGMENT_HINTS.get(answer_format, ""),
            answer_hint=ANSWER_HINTS.get(answer_format, ""),
        )
        
        try:
            result = self.qwen.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": reasoning_prompt},
                ],
                temperature=0.1, max_tokens=2048, timeout=120,
            )
            raw_response = result["content"]
        except Exception as e:
            raw_response = ""
        
        # 提取答案
        answer = extract_answer_from_response(raw_response, answer_format)
        
        # Step 5: Self-Critique（仅多选题且 >= 2 个选项被选）
        if answer_format == "multi" and answer and len(answer) >= 2:
            answer = self._self_critique(q_text, options, answer, raw_response)
        
        # 后处理
        answer = self._post_process(answer, answer_format)
        
        # 保存 CoT 轨迹
        self.cot_trails.append({
            "qid": qid,
            "domain": domain,
            "keywords_count": len(keywords),
            "bm25_count": len(bm25_results),
            "kw_count": len(kw_results),
            "merged_count": len(merged),
            "evidence_chars": len(evidence_text),
            "answer": answer,
        })
        
        return {"qid": qid, "answer": answer, "raw_response": raw_response}
    
    def _self_critique(self, question: str, options: dict, answer: str, raw: str) -> str:
        """Self-Critique 二次校验（多选题专用，只允许删除不允许添加）"""
        try:
            prompt = SELF_CRITIQUE_PROMPT.format(answer=answer, raw_response=raw[:1500])
            result = self.qwen.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=512, timeout=60,
            )
            corrected = extract_answer_from_response(result["content"], "multi")
            # 只允许删除，不允许添加
            if corrected and set(corrected).issubset(set(answer)):
                return corrected
            return answer
        except:
            return answer
    
    def _post_process(self, answer: str, answer_format: str) -> str:
        """强格式校验"""
        if not answer:
            return ""
        
        valid_letters = set("ABCD")
        
        if answer_format == "mcq":
            for c in answer:
                if c in valid_letters:
                    return c
            return "A"
        elif answer_format == "tf":
            if "A" in answer:
                return "A"
            if "B" in answer:
                return "B"
            return "A"
        elif answer_format == "multi":
            letters = sorted(set(c for c in answer if c in valid_letters))
            if len(letters) > 3:
                letters = letters[:3]
            return "".join(letters) if letters else ""
        
        return answer
    
    def save_cot_trails(self):
        """保存 CoT 轨迹"""
        path = os.path.join(RESULTS_DIR, "eval_results_full.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)
