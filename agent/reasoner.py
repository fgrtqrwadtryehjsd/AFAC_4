"""推理引擎 V5+ — 全面改进版

整合技术：
1. RRF 倒数排名融合 — 多路召回去重合并
2. Lost-in-the-Middle 缓解 — 重要信息放首尾
3. Self-Critique 二次校验 — 多选题反事实检查
4. 思维链推理 — 三步 CoT
5. 答案后处理 — 强格式校验
6. CoT 轨迹保存 — 供复盘分析
"""
import os
import json
import re
import time
from agent.config import RESULTS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex, ContextWindowOptimizer
from agent.memory_compressor import MemoryManager
from agent.postprocessor import extract_answer_from_response


# ============ RRF 倒数排名融合 ============

def reciprocal_rank_fusion(result_lists: list, k: int = 60, weights: list = None) -> list:
    """RRF: 多路召回结果融合
    
    Args:
        result_lists: 每路召回的 chunks 列表
        k: RRF 平滑常数（默认60）
        weights: 每路权重（条款定位>选项检索>BM25）
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    doc_scores = {}  # text_key -> {"score": float, "chunk": dict}
    
    for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
        for rank, chunk in enumerate(results):
            text_key = chunk.get("text", "")[:100]  # 去重键
            rrf_score = weight / (k + rank + 1)
            
            if text_key in doc_scores:
                doc_scores[text_key]["score"] += rrf_score
            else:
                doc_scores[text_key] = {"score": rrf_score, "chunk": chunk}
    
    # 按 RRF 分数排序
    sorted_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["chunk"] for item in sorted_results]


# ============ Lost-in-the-Middle 缓解 ============

def arrange_context_for_llm(chunks: list) -> list:
    """将最重要的证据放在 Prompt 的最头部和最尾部
    大模型对中间信息注意力最弱（Lost in the Middle）
    """
    if len(chunks) <= 2:
        return chunks
    
    # 最重要(第一个)放最前，次重要(第二个)放最后，其余放中间
    head = [chunks[0]] if chunks else []
    tail = [chunks[1]] if len(chunks) > 1 else []
    middle = chunks[2:] if len(chunks) > 2 else []
    
    return head + middle + tail


# ============ Prompt Engineering ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": """你是一位资深保险精算师和条款解读专家。
严格依据文档原文和摘要回答问题，不用常识推断。

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

# ============ 思维链推理模板 ============

COT_REASONING_PROMPT = """## 任务
基于以下文档记忆和证据，准确回答问题。

## 文档记忆（压缩摘要）
{memory}

## 检索证据
{evidence}

## 问题
{question}

## 选项
{options}

## 答题要求
1. 严格依据文档记忆和证据判断，不用常识和外部知识
2. {format_specific}

请按思维链逐步推理：

**步骤1：定位关键信息**
从文档记忆和证据中，找到与问题直接相关的关键事实、条款、数字。

**步骤2：逐选项分析**
对每个选项，判断文档是否有明确支持：
- 选项A：[有明确支持/无明确支持] — 引用：[原文关键语句]
- 选项B：[有明确支持/无明确支持] — 引用：[原文关键语句]
- 选项C：[有明确支持/无明确支持] — 引用：[原文关键语句]
- 选项D：[有明确支持/无明确支持] — 引用：[原文关键语句]

**步骤3：综合判断**
{judgment_hint}

最终答案：{answer_hint}"""

# ============ Self-Critique 二次校验（多选题专用） ============

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


# ============ 推理 Agent ============

class ReasoningAgent:
    """推理 Agent V5+
    
    流程：
    1. 压缩记忆获取
    2. RAG 三路召回 → RRF 融合
    3. 上下文窗口优化 + Lost-in-the-Middle 重排
    4. 思维链分步推理
    5. Self-Critique 二次校验（多选题）
    6. 答案后处理
    7. 保存 CoT 轨迹
    """

    def __init__(self, qwen_client: QwenClient, doc_index: DocumentIndex,
                 memory_manager: MemoryManager, token_budget: int = 4_500_000):
        self.qwen = qwen_client
        self.index = doc_index
        self.memory = memory_manager
        self.context_optimizer = ContextWindowOptimizer(max_context_chars=10000)
        self.token_budget = token_budget
        self.cot_trails = []  # 保存推理轨迹

    def tokens_remaining(self) -> int:
        return max(0, self.token_budget - self.qwen.get_token_stats()["total_tokens"])

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        question_text = question["question"]
        options = question["options"]
        answer_format = question["answer_format"]
        doc_ids = question.get("doc_ids", [])

        print(f"  {qid} ({domain}/{answer_format})", end=" ")

        # Step 1: 压缩记忆
        compressed_memory = self.memory.get_context(doc_ids, question_text, domain, max_chars=8000)

        # Step 2: RAG 三路召回 + RRF 融合
        merged_chunks = self._multi_recall(question_text, options, doc_ids)

        # Step 3: 上下文窗口优化 + Lost-in-the-Middle 重排
        evidence = self._build_evidence(compressed_memory, merged_chunks)

        # Step 4: 思维链推理
        answer, raw = self._cot_reason(domain, question_text, options, compressed_memory, evidence, answer_format)

        # Step 5: Self-Critique（多选题）
        if answer_format == "multi" and answer:
            answer, critique_raw = self._self_critique(question_text, options, answer, raw)
        else:
            critique_raw = ""

        # Step 6: 答案后处理
        if not answer:
            answer = extract_answer_from_response(raw, answer_format)
        answer = self._post_process(answer, answer_format, options)

        # Step 7: 保存 CoT 轨迹
        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer_format": answer_format,
            "question": question_text, "options": options, "doc_ids": doc_ids,
            "answer": answer, "raw_cot": raw, "self_critique": critique_raw,
            "evidence_count": len(merged_chunks),
            "memory_length": len(compressed_memory),
        })

        print(f"→ {answer}")
        return {"qid": qid, "answer": answer, "raw_response": raw}

    def _multi_recall(self, question: str, options: dict, doc_ids: list) -> list:
        """三路召回 + RRF 融合"""
        from agent.clause_locator import extract_clause_refs, locate_clauses_in_doc
        
        # 路线1: BM25 主查询
        bm25_results = self.index.search_bm25(question, top_k=10, doc_ids=doc_ids if doc_ids else None)
        
        # 路线2: 选项检索
        option_results = []
        for opt_key, opt_text in options.items():
            opt_res = self.index.search_bm25(f"{question} {opt_text}", top_k=3, doc_ids=doc_ids if doc_ids else None)
            option_results.extend(opt_res)
        
        # 路线3: 条款精准定位
        clause_results = []
        all_text = question + " " + " ".join(str(v) for v in options.values())
        clause_refs = extract_clause_refs(all_text, options)
        if clause_refs and doc_ids:
            for doc_id in doc_ids:
                full_text = self.index.get_doc_full_text(doc_id)
                if full_text:
                    located = locate_clauses_in_doc(clause_refs, full_text)
                    for ref, text in located.items():
                        clause_results.append({
                            "doc_id": doc_id, "text": text,
                            "chunk_type": "clause", "clause_ref": ref,
                            "score": 100  # 条款精准定位最高分
                        })
        
        # RRF 融合：条款定位权重最高
        merged = reciprocal_rank_fusion(
            [clause_results, bm25_results, option_results],
            weights=[3.0, 1.0, 1.0]  # 条款3x权重
        )
        return merged[:15]

    def _build_evidence(self, compressed_memory: str, chunks: list) -> str:
        """构建证据文本，Lost-in-the-Middle 重排"""
        # 分离条款和普通证据
        clause_chunks = [c for c in chunks if c.get("chunk_type") == "clause"]
        other_chunks = [c for c in chunks if c.get("chunk_type") != "clause"]
        
        # Lost-in-the-Middle: 重要信息放首尾
        arranged_other = arrange_context_for_llm(other_chunks)
        
        # 构建证据文本
        parts = []
        
        # 条款精准定位放最前（最重要）
        for c in clause_chunks:
            parts.append(f"### 条款定位：{c.get('clause_ref','')}\n{c['text'][:1000]}")
        
        # 压缩记忆
        if compressed_memory:
            parts.append(f"### 文档摘要\n{compressed_memory[:4000]}")
        
        # RRF 排序后的 BM25 证据
        for i, c in enumerate(arranged_other):
            src = c.get("doc_id", "?")
            parts.append(f"### 证据{i+1}({src})\n{c['text'][:600]}")
        
        total = "\n\n".join(parts)
        # 限制总量
        if len(total) > 10000:
            total = total[:10000]
        return total

    def _cot_reason(self, domain: str, question: str, options: dict,
                    memory: str, evidence: str, answer_format: str) -> tuple:
        """思维链分步推理"""
        sys_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "严格依据文档证据回答。")

        user_prompt = COT_REASONING_PROMPT.format(
            memory=memory or "（无压缩记忆）",
            evidence=evidence or "（无检索证据）",
            question=question,
            options=format_options(options),
            format_specific=FORMAT_SPECIFIC.get(answer_format, "选择正确选项"),
            judgment_hint=JUDGMENT_HINTS.get(answer_format, "综合判断"),
            answer_hint=ANSWER_HINTS.get(answer_format, "一个选项"),
        )

        result = self.qwen.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1, max_tokens=2048, timeout=120,
        )
        answer = extract_answer_from_response(result["content"], answer_format)
        return answer, result["content"]

    def _self_critique(self, question: str, options: dict, answer: str, raw: str) -> tuple:
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
                return corrected, result["content"]
            return answer, ""
        except:
            return answer, ""

    def _post_process(self, answer: str, answer_format: str, options: dict) -> str:
        """答案后处理：强格式校验"""
        if not answer:
            return answer

        valid_letters = set(sorted(options.keys()))

        if answer_format == "mcq":
            answer = "".join(c for c in answer if c in valid_letters)
            return answer[:1] if answer else ""

        elif answer_format == "tf":
            answer = answer.upper()
            answer = answer.replace("正确", "A").replace("错误", "B").replace("对", "A").replace("错", "B")
            if "A" in answer and "B" not in answer:
                return "A"
            elif "B" in answer:
                return "B"
            return "A"

        elif answer_format == "multi":
            letters = sorted(set(c for c in answer if c in valid_letters))
            # 限制最多3个选项（4个全选几乎不可能正确）
            if len(letters) > 3:
                letters = letters[:3]
            return "".join(letters) if letters else ""

        return answer

    def save_cot_trails(self, filepath: str = None):
        """保存 CoT 推理轨迹"""
        filepath = filepath or os.path.join(RESULTS_DIR, "eval_results_full.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)
        print(f"  CoT 轨迹已保存: {filepath} ({len(self.cot_trails)} 条)")
