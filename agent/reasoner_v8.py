"""V8: 精确证据推理 Agent (Precise Evidence Reasoning Agent)

核心技术突破（基于 V7.1 30.28分的根因分析）：

1. 选项去偏 (Option Debiasing):
   - 打乱选项顺序（随机排列ABCD），消除模型的A-bias
   - 51/69单选题选A → 真实分布应该更均匀
   - 推理后还原选项标签

2. 全文证据策略 (Full-Document Evidence):
   - 利用剩余3.2M Token预算，大幅增加证据量
   - qwen-plus 上下文窗口131K tokens ≈ 200K chars
   - 短文档(<50K): 喂全文
   - 中文档(50K-150K): 喂全文（仍在窗口内）
   - 长文档(>150K): Card引导的智能分段 + 高相关度段落

3. 两阶段推理 (Two-Stage Reasoning):
   - Stage 1: 证据扫描 → 逐选项快速判断（有支持/无支持/不确定）
   - Stage 2: 对不确定选项，定向搜索原文再验证
   - 避免"全选"和"漏选"两个极端
"""
import os
import json
import re
import time
import random
from collections import defaultdict
from agent.config import RESULTS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.postprocessor import extract_answer_from_response


# ============ 选项去偏 ============

def shuffle_options(options: dict) -> tuple:
    """打乱选项顺序，返回 (shuffled_options, mapping)
    
    mapping: {原始标签: 打乱后标签}
    例如: {"A": "C", "B": "A", "C": "D", "D": "B"}
    意味着原始选项A现在显示为C，原始选项B现在显示为A
    """
    labels = list(options.keys())
    texts = list(options.values())
    
    # 保留原始标签集，打乱文本顺序
    shuffled_texts = texts.copy()
    random.shuffle(shuffled_texts)
    
    shuffled = {}
    reverse_map = {}  # 打乱后标签 → 原始标签
    for i, label in enumerate(labels):
        shuffled[label] = shuffled_texts[i]
        # 找到这个文本在原始options中的标签
        for orig_label, orig_text in options.items():
            if orig_text == shuffled_texts[i]:
                reverse_map[label] = orig_label
                break
    
    forward_map = {}  # 原始标签 → 打乱后标签
    for new_label, orig_label in reverse_map.items():
        forward_map[orig_label] = new_label
    
    return shuffled, forward_map, reverse_map


def unshuffle_answer(answer: str, reverse_map: dict) -> str:
    """将打乱后的答案还原为原始标签"""
    return "".join(sorted(reverse_map.get(c, c) for c in answer))


# ============ 基于规则的结构化Card ============

CLAUSE_PATTERN = re.compile(r'第[一二三四五六七八九十百千万零〇\d]+[条章节款项]', re.UNICODE)
MONEY_PATTERN = re.compile(r'[\d,.]+[万亿]?[元美元人民币]')
PERCENT_PATTERN = re.compile(r'[\d.]+%|[\d.]+百分之')
DATE_PATTERN = re.compile(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}')
ENTITY_PATTERN = re.compile(r'[\u4e00-\u9fff]{2,20}(?:股份|有限|集团|公司|银行|保险|证券|基金|信托)')

CN_NUM_MAP = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '百': 100, '千': 1000, '万': 10000,
}

def cn_to_num(cn: str) -> int:
    if cn.isdigit():
        return int(cn)
    result = 0
    current = 0
    for char in cn:
        if char in CN_NUM_MAP:
            val = CN_NUM_MAP[char]
            if val >= 10:
                if current == 0:
                    current = 1
                result += current * val
                current = 0
            else:
                current = current * 10 + val if current >= 10 else current * 10 + val
    return result + current


def extract_structured_card(doc_id: str, full_text: str) -> dict:
    """基于规则的结构化Card提取（零Token成本）"""
    if not full_text:
        return {"doc_id": doc_id, "char_count": 0}
    
    card = {
        "doc_id": doc_id,
        "char_count": len(full_text),
        "clauses": [],
        "metrics": [],
        "entities": [],
        "section_titles": [],
        "key_terms": [],
    }
    
    seen_refs = set()
    for m in CLAUSE_PATTERN.finditer(full_text):
        ref = m.group()
        pos = m.start()
        if ref not in seen_refs:
            seen_refs.add(ref)
            next_clause = CLAUSE_PATTERN.search(full_text, pos + len(ref))
            end = next_clause.start() if next_clause else min(pos + 200, len(full_text))
            preview = full_text[pos:end].strip()[:200]
            card["clauses"].append({"ref": ref, "position": pos, "content_preview": preview})
    
    for m in MONEY_PATTERN.finditer(full_text):
        value = m.group()
        ctx_start = max(0, m.start() - 30)
        ctx_end = min(len(full_text), m.end() + 30)
        context = full_text[ctx_start:ctx_end].strip()
        card["metrics"].append({"type": "money", "value": value, "context": context})
    
    for m in PERCENT_PATTERN.finditer(full_text):
        value = m.group()
        ctx_start = max(0, m.start() - 30)
        ctx_end = min(len(full_text), m.end() + 30)
        context = full_text[ctx_start:ctx_end].strip()
        card["metrics"].append({"type": "percent", "value": value, "context": context})
    
    entities = set()
    for m in ENTITY_PATTERN.finditer(full_text):
        entity = m.group()
        if len(entity) >= 4:
            entities.add(entity)
    card["entities"] = list(entities)
    
    section_pattern = re.compile(r'第[一二三四五六七八九十百千万零〇\d]+[章节]\s*[^\n]{2,30}')
    for m in section_pattern.finditer(full_text):
        title = m.group().strip()
        card["section_titles"].append({"title": title, "position": m.start()})
    
    financial_terms = [
        "特别决议", "普通决议", "累积投票", "独立董事", "股东大会", "董事会",
        "监事会", "保荐人", "受托管理人", "违约事件", "加速到期", "信用评级",
        "主体评级", "债项评级", "偿付顺序", "保证金", "担保人", "连带责任",
        "不可撤销", "可撤销", "持有人会议", "回售", "赎回", "票面利率",
        "到期日", "发行规模", "募集资金", "已交保费", "现金价值", "账户价值",
        "保险责任", "免责条款", "等待期", "犹豫期", "退保费用", "身故保险金",
        "特别处理", "退市", "暂停上市", "恢复上市", "要约收购", "协议收购",
        "权益分派", "利润分配", "股息红利", "送股", "转增", "配股",
        "投资收益", "公允价值", "减值准备", "坏账准备", "折旧摊销",
        "经营活动", "投资活动", "筹资活动", "现金流量", "净利润",
        "营业收入", "营业成本", "毛利率", "净利率", "资产负债率",
        "流动比率", "速动比率", "应收账款", "存货周转",
    ]
    term_candidates = defaultdict(int)
    for term in financial_terms:
        count = full_text.count(term)
        if count > 0:
            term_candidates[term] = count
    card["key_terms"] = [
        {"term": t, "count": c} for t, c in 
        sorted(term_candidates.items(), key=lambda x: -x[1])[:30]
    ]
    
    return card


# ============ Agent 记忆层 ============

class AgentMemory:
    def __init__(self, doc_index: DocumentIndex):
        self.doc_index = doc_index
        self.cards = {}
        self.doc_access_count = defaultdict(int)
        self.questions_answered = 0
        self.card_build_time = 0
    
    def get_card(self, doc_id: str) -> dict:
        if doc_id in self.cards:
            self.doc_access_count[doc_id] += 1
            return self.cards[doc_id]
        t0 = time.time()
        full_text = self.doc_index.get_doc_full_text(doc_id)
        card = extract_structured_card(doc_id, full_text)
        self.card_build_time += time.time() - t0
        self.cards[doc_id] = card
        self.doc_access_count[doc_id] = 1
        return card
    
    def get_card_match_hints(self, question: str, options: dict, doc_ids: list) -> dict:
        hints = {"clause_refs": [], "metric_terms": [], "entity_terms": [], "key_term_hits": []}
        full_query = question + " " + " ".join(options.values())
        for m in CLAUSE_PATTERN.finditer(full_query):
            hints["clause_refs"].append(m.group())
        for doc_id in doc_ids:
            card = self.get_card(doc_id)
            for entity in card.get("entities", []):
                if entity in full_query:
                    hints["entity_terms"].append(entity)
            for metric in card.get("metrics", []):
                value = metric.get("value", "")
                if value in full_query:
                    hints["metric_terms"].append(value)
            for kt in card.get("key_terms", []):
                term = kt.get("term", "")
                if term in full_query:
                    hints["key_term_hits"].append(term)
        return hints
    
    def locate_clause_text(self, doc_id: str, clause_ref: str, max_len: int = 2000) -> str:
        full_text = self.doc_index.get_doc_full_text(doc_id)
        if not full_text:
            return ""
        pos = full_text.find(clause_ref)
        if pos < 0:
            return ""
        next_clause = CLAUSE_PATTERN.search(full_text, pos + len(clause_ref))
        end = next_clause.start() if next_clause else len(full_text)
        return full_text[pos:min(end, pos + max_len)].strip()


# ============ 关键词检索 ============

def extract_query_keywords(question: str, options: dict) -> list:
    keywords = set()
    stopwords = {'的','了','在','是','和','与','或','及','等','对','为','从','中','不','有','这',
                 '那','也','都','就','而','但','如','其','以','所','可','将','被','让','给','比',
                 '到','于','以下','关于','上述','下列','哪些','哪个','是否','属于',
                 '以下哪些','下列哪些','描述','说法','结论','正确','准确','成立','判断'}
    for pattern in [r'[\u4e00-\u9fff]{2,8}', r'\d+\.?\d*%?', r'[A-Z]{2,}']:
        for m in re.finditer(pattern, question):
            w = m.group()
            if w not in stopwords and len(w) >= 2:
                keywords.add(w)
    for opt_text in options.values():
        for pattern in [r'[\u4e00-\u9fff]{2,8}', r'\d+\.?\d*%?', r'[A-Z]{2,}']:
            for m in re.finditer(pattern, opt_text):
                w = m.group()
                if w not in stopwords and len(w) >= 2:
                    keywords.add(w)
    return list(keywords)


def keyword_match_score(keywords: list, text: str) -> float:
    score = 0.0
    for kw in keywords:
        count = text.count(kw)
        if count > 0:
            weight = max(1, len(kw) - 1) if len(kw) <= 3 else len(kw)
            score += count * weight
    return score


def rrf_fuse(bm25_results: list, kw_results: list, k: int = 60,
             bm25_weight: float = 1.0, kw_weight: float = 1.5) -> list:
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


# ============ Prompt ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": "你是保险条款分析专家。严格依据文档原文判断每个选项。只有明确原文条款支持才选。",
    "regulatory": "你是金融监管合规专家。严格依据法规原文判断每个选项。区分'应当'(强制)vs'可以'(授权)。只有明确原文条款支持才选。",
    "financial_contracts": "你是金融合同分析专家。严格依据合同条款原文判断每个选项。主体评级≠债项评级。只有明确原文条款支持才选。",
    "financial_reports": "你是财务报表分析专家。严格依据年报精确数字判断每个选项。同比=去年同期。只有明确原文数据支持才选。",
    "research": "你是行业研报分析专家。严格依据研报数据判断每个选项。'预计'≠'实际'。只有明确原文数据支持才选。",
}


STAGE1_PROMPT = """## 任务
严格依据文档证据，逐选项判断是否正确。

## 文档证据
{evidence}

## 问题
{question}

## 选项（注意：选项顺序已随机打乱）
{options}

## 逐选项严格验证

对每个选项，你必须：
1. 搜索文档中与该选项直接相关的原文
2. 引用原文的精确语句（不能改写或概括）
3. 基于原文精确用语判断是否完全支持选项的完整表述

**选项A**：{option_a}
原文精确引用：
精确比对：选项表述 vs 原文表述 是否完全一致？
判断：✅ 明确支持 / ❌ 不支持 / ⚠️ 部分相关但不完全支持

**选项B**：{option_b}
原文精确引用：
精确比对：
判断：✅ 明确支持 / ❌ 不支持 / ⚠️ 部分相关但不完全支持

**选项C**：{option_c}
原文精确引用：
精确比对：
判断：✅ 明确支持 / ❌ 不支持 / ⚠️ 部分相关但不完全支持

**选项D**：{option_d}
原文精确引用：
精确比对：
判断：✅ 明确支持 / ❌ 不支持 / ⚠️ 部分相关但不完全支持

## 最终答案
只选择 ✅明确支持 的选项。⚠️部分相关不算支持。
{answer_hint}"""


SELF_CRITIQUE_PROMPT = """多选题 {answer} 的严格二次校验。

对每个已选选项，必须确认文档中有明确的、直接的原文条款支持。
- 间接推断 ≠ 明确支持
- 模糊相关 ≠ 明确支持  
- 常识判断 ≠ 明确支持
- 如果对某选项找不到明确原文条款，必须删除

最终答案（只能删除，不能添加）："""


# ============ V8 Agent ============

class ReasoningAgentV8:
    """V8: 精确证据推理 Agent
    
    核心突破：
    1. 选项去偏 — 打乱ABCD顺序消除A-bias
    2. 全文证据 — 大幅利用Token预算(5M只用1.8M)
    3. 两阶段推理 — 先扫描后验证
    """
    
    def __init__(self, qwen: QwenClient, doc_index: DocumentIndex,
                 token_budget: int = 5_000_000, model: str = "qwen-plus"):
        self.qwen = qwen
        self.doc_index = doc_index
        self.token_budget = token_budget
        self.memory = AgentMemory(doc_index)
        self.cot_trails = []
        self._shuffle_count = 0
        self._shuffle_changed_answer = 0
    
    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])
        
        # === Phase 1: 选项去偏 ===
        shuffled_opts, fwd_map, rev_map = shuffle_options(options)
        self._shuffle_count += 1
        
        # === Phase 2: Card匹配 ===
        card_hints = self.memory.get_card_match_hints(q_text, options, doc_ids)
        
        # === Phase 3: 证据组装 ===
        total_doc_chars = sum(
            self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        
        # 全文策略：根据Token预算和题型智能决定证据量
        # V7.1用30K证据→1.83M token(30.28分)
        # V8用180K证据→4.85M token只答39题（Token耗尽）
        # 平衡点：80K chars证据 → 足够覆盖关键信息，且100题不超预算
        max_context_chars = 40000  # 40K chars ≈ 27K tokens
        
        if total_doc_chars <= max_context_chars:
            # 全部文档可以放入 → 喂全文
            evidence_text = ""
            for doc_id in doc_ids:
                ft = self.doc_index.get_doc_full_text(doc_id)
                if ft:
                    evidence_text += f"\n=== 文档 {doc_id} (全文 {len(ft)}字) ===\n{ft}\n"
        else:
            # 文档总量超过窗口 → 智能检索
            evidence_text = self._smart_retrieve(
                q_text, shuffled_opts, doc_ids, card_hints, max_context_chars)
        
        # === Phase 4: CoT推理（打乱后的选项）===
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")
        
        answer_hint = {
            "mcq": "一个大写字母(A/B/C/D)，只选有✅明确支持的",
            "tf": "A或B，A=正确B=错误",
            "multi": "多个大写字母按字母序(如ABC)，只包含有✅明确支持的选项，⚠️不算",
        }.get(answer_format, "")
        
        stage1_prompt = STAGE1_PROMPT.format(
            evidence=evidence_text,
            question=q_text,
            options="\n".join(f"{k}. {shuffled_opts[k]}" for k in sorted(shuffled_opts.keys())),
            option_a=shuffled_opts.get("A", ""),
            option_b=shuffled_opts.get("B", ""),
            option_c=shuffled_opts.get("C", ""),
            option_d=shuffled_opts.get("D", ""),
            answer_hint=answer_hint,
        )
        
        try:
            result = self.qwen.chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": stage1_prompt}],
                temperature=0.1, max_tokens=4096, timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""
        
        # 提取打乱后的答案
        shuffled_answer = extract_answer_from_response(raw_response, answer_format)
        
        # === Phase 5: 还原选项标签 ===
        original_answer = unshuffle_answer(shuffled_answer, rev_map)
        
        # === Phase 6: Self-Critique（多选题，用原始标签）===
        if answer_format == "multi" and original_answer and len(original_answer) >= 2:
            try:
                critique_result = self.qwen.chat(
                    [{"role": "user", "content": SELF_CRITIQUE_PROMPT.format(answer=original_answer)}],
                    temperature=0.0, max_tokens=256, timeout=60,
                )
                corrected = extract_answer_from_response(critique_result["content"], "multi")
                if corrected and set(corrected).issubset(set(original_answer)):
                    original_answer = corrected
            except:
                pass
        
        original_answer = self._post_process(original_answer, answer_format)
        self.memory.questions_answered += 1
        
        # 记录选项打乱是否改变了答案（vs不打的对比）
        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": original_answer,
            "shuffled_answer": shuffled_answer,
            "shuffle_map": fwd_map,
            "clause_refs_found": len(card_hints["clause_refs"]),
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
            "card_cache_total": sum(self.memory.doc_access_count.values()),
        })
        
        return {"qid": qid, "answer": original_answer, "raw_response": raw_response}
    
    def _smart_retrieve(self, q_text: str, options: dict, doc_ids: list,
                         card_hints: dict, max_chars: int) -> str:
        """智能检索：对于超长文档，使用多路召回+Card引导"""
        # BM25 主查询
        bm25_results = self.doc_index.search_bm25(q_text, top_k=30, doc_ids=doc_ids)
        
        # 选项独立检索
        seen = set(c.get("text", "")[:80] for c in bm25_results)
        for opt_key, opt_text in sorted(options.items()):
            opt_res = self.doc_index.search_bm25(
                f"{q_text} {opt_text}", top_k=3, doc_ids=doc_ids)
            for c in opt_res:
                key = c.get("text", "")[:80]
                if key not in seen:
                    seen.add(key)
                    bm25_results.append(c)
        
        # Card引导的关键词检索
        query_keywords = extract_query_keywords(q_text, options)
        search_keywords = list(set(
            query_keywords + card_hints["entity_terms"] + card_hints["key_term_hits"]
        ))
        all_doc_chunks = self.doc_index.get_chunks_by_doc_ids(doc_ids)
        kw_scored = []
        for chunk in all_doc_chunks:
            text = chunk.get("text", "")
            score = keyword_match_score(search_keywords, text)
            if score > 0:
                kw_scored.append((score, chunk))
        kw_scored.sort(key=lambda x: -x[0])
        kw_results = [chunk for _, chunk in kw_scored[:60]]
        
        # 条款精准定位
        clause_evidence = []
        for doc_id in doc_ids:
            for clause_ref in card_hints["clause_refs"]:
                clause_text = self.memory.locate_clause_text(doc_id, clause_ref)
                if clause_text:
                    clause_evidence.append({
                        "doc_id": doc_id, "text": clause_text,
                        "chunk_type": "clause_precise", "clause_ref": clause_ref,
                    })
        
        # RRF融合
        merged = rrf_fuse(bm25_results, kw_results)
        final_evidence = clause_evidence + merged[:60]
        
        evidence_text = ""
        for chunk in final_evidence:
            doc_id = chunk.get("doc_id", "")
            text = chunk.get("text", "")
            ref = chunk.get("clause_ref", "")
            label = f"文档 {doc_id}" + (f" [{ref}]" if ref else "")
            evidence_text += f"\n--- 来自{label} ---\n{text}\n"
            if len(evidence_text) > max_chars:
                break
        
        if len(evidence_text) > max_chars:
            evidence_text = evidence_text[:max_chars] + "\n[...证据已截断...]"
        
        return evidence_text
    
    def _post_process(self, answer: str, answer_format: str) -> str:
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
        path = os.path.join(RESULTS_DIR, "eval_results_full.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)
        
        card_path = os.path.join(RESULTS_DIR, "document_cards.json")
        cards_data = {doc_id: card for doc_id, card in self.memory.cards.items()}
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(cards_data, f, ensure_ascii=False, indent=2)
