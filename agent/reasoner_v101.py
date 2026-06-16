"""V10.1: Verification Agent — Token优化版

V10问题: 4次独立Qwen验证调用 → 4.66M Token
V10.1方案: 逐选项检索证据 → 1次统一CoT验证 → 答案

关键优化:
1. 逐选项检索证据（保留！提升证据质量）
2. 1次统一CoT验证所有选项（4次→1次，Token降4倍）
3. 短文档(≤80K)全文输入
4. 长文档每选项8K证据 → 合并后40K证据
5. 消除position bias的证据呈现方式
"""
import os
import json
import re
import time
from collections import defaultdict
from agent.config import RESULTS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.vector_indexer import VectorIndexer
from agent.postprocessor import extract_answer_from_response

# ============ 复用工具 ============

CN_NUM_MAP = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '百': 100, '千': 1000, '万': 10000,
}
CLAUSE_PATTERN = re.compile(r'第[一二三四五六七八九十百千万零〇\d]+[条章节款项]', re.UNICODE)
MONEY_PATTERN = re.compile(r'[\d,.]+[万亿]?[元美元人民币]')
PERCENT_PATTERN = re.compile(r'[\d.]+%|[\d.]+百分之')
DATE_PATTERN = re.compile(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}')
ENTITY_PATTERN = re.compile(r'[\u4e00-\u9fff]{2,20}(?:股份|有限|集团|公司|银行|保险|证券|基金|信托)')


def extract_structured_card(doc_id: str, full_text: str) -> dict:
    if not full_text:
        return {"doc_id": doc_id, "char_count": 0}
    card = {
        "doc_id": doc_id, "char_count": len(full_text),
        "clauses": [], "metrics": [], "entities": [],
        "section_titles": [], "key_terms": [],
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
        card["metrics"].append({"type": "money", "value": value, "context": full_text[ctx_start:ctx_end].strip()})
    for m in PERCENT_PATTERN.finditer(full_text):
        value = m.group()
        ctx_start = max(0, m.start() - 30)
        ctx_end = min(len(full_text), m.end() + 30)
        card["metrics"].append({"type": "percent", "value": value, "context": full_text[ctx_start:ctx_end].strip()})
    for m in DATE_PATTERN.finditer(full_text):
        value = m.group()
        ctx_start = max(0, m.start() - 20)
        ctx_end = min(len(full_text), m.end() + 20)
        card["metrics"].append({"type": "date", "value": value, "context": full_text[ctx_start:ctx_end].strip()})
    entities = set()
    for m in ENTITY_PATTERN.finditer(full_text):
        entity = m.group()
        if len(entity) >= 4:
            entities.add(entity)
    card["entities"] = list(entities)
    section_pattern = re.compile(r'第[一二三四五六七八九十百千万零〇\d]+[章节]\s*[^\n]{2,30}')
    for m in section_pattern.finditer(full_text):
        card["section_titles"].append({"title": m.group().strip(), "position": m.start()})
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


def extract_query_keywords(text: str) -> list:
    keywords = set()
    stopwords = {'的','了','在','是','和','与','或','及','等','对','为','从','中','不','有','这',
                 '那','也','都','就','而','但','如','其','以','所','可','将','被','让','给','比',
                 '到','于','以下','关于','上述','下列','哪些','哪个','是否','属于',
                 '以下哪些','下列哪些','描述','说法','结论','正确','准确','成立','判断'}
    for pattern in [r'[\u4e00-\u9fff]{2,8}', r'\d+\.?\d*%?', r'[A-Z]{2,}']:
        for m in re.finditer(pattern, text):
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


def compress_whitespace(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


# ============ 领域Prompt ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": "你是保险条款审查员。严格依据文档原文逐选项判断。身故保险金≠已交保费≠现金价值≠账户价值。退保金额=现金价值-退保费用。只有明确原文支持才判定为正确。",

    "regulatory": "你是金融法规审查员。严格依据法规原文逐选项判断。\"应当\"/\"必须\"/\"不得\"=强制性；\"可以\"=授权性。\"经股东大会审议\"≠\"经特别决议通过\"。只有明确原文支持才判定为正确。",

    "financial_contracts": "你是金融合同审查员。严格依据合同条款原文逐选项判断。主体评级≠债项评级。违约事件须满足合同明确定义。只有明确原文支持才判定为正确。",

    "financial_reports": "你是财务报表审查员。严格依据年报精确数字逐选项判断。同比=去年同期；环比=上期。\"拟派发\"≠\"已派发\"。只有明确原文数据支持才判定为正确。",

    "research": "你是行业研报审查员。严格依据研报数据逐选项判断。\"预计/预期\"≠\"实际\"。只有明确原文数据支持才判定为正确。",
}


# ============ 统一验证Prompt（1次调用验证所有选项）============

UNIFIED_VERIFY_PROMPT = """## 任务
你是证据审查员。严格依据文档证据，对每个选项独立判断是否有明确原文支持。

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 逐选项审查

对每个选项，你必须：
1. 从文档证据中搜索与该选项直接相关的原文
2. 引用原文的精确语句（不能改写或概括）
3. 判断原文是否直接、完整地支持该选项

**选项A**：{option_a}
原文引用：
分析：原文是否直接支持选项A的完整表述？注意"应当"≠"可以"等精确用语
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

**选项B**：{option_b}
原文引用：
分析：原文是否直接支持选项B的完整表述？
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

**选项C**：{option_c}
原文引用：
分析：原文是否直接支持选项C的完整表述？
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

**选项D**：{option_d}
原文引用：
分析：原文是否直接支持选项D的完整表述？
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

最终答案：{answer_hint}"""


# ============ Memory Agent ============

class AgentMemory:
    def __init__(self, doc_index: DocumentIndex):
        self.doc_index = doc_index
        self.cards = {}
        self.doc_access_count = defaultdict(int)
        self.card_build_time = 0
        self.questions_answered = 0

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
        hints = {"clause_refs": [], "entity_terms": [], "key_term_hits": []}
        full_query = question + " " + " ".join(options.values())
        for m in CLAUSE_PATTERN.finditer(full_query):
            hints["clause_refs"].append(m.group())
        for doc_id in doc_ids:
            card = self.get_card(doc_id)
            for entity in card.get("entities", []):
                if entity in full_query:
                    hints["entity_terms"].append(entity)
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


# ============ V10.1 Verification Agent ============

class ReasoningAgentV101:
    """V10.1: 逐选项检索 + 1次统一验证 (Token优化版)

    V10: 4次独立验证 → 4.66M Token ❌
    V10.1: 逐选项检索证据 → 1次统一CoT验证 → 答案 ✅

    流程：
    1. 短文档(≤80K): 全文输入
    2. 长文档: 逐选项检索证据 → 合并证据
    3. 1次统一CoT验证所有选项
    4. Self-Critique（多选题）
    """

    FULL_DOC_THRESHOLD = 80000

    EVIDENCE_PER_OPTION = {
        "mcq": 6000,
        "tf": 5000,
        "multi": 8000,
    }

    TOTAL_EVIDENCE_LIMIT = {
        "mcq": 20000,     # 4选项×6K ≈ 24K，限20K
        "tf": 10000,      # 2选项×5K ≈ 10K
        "multi": 35000,   # 4选项×8K ≈ 32K，限35K
    }

    def __init__(self, qwen: QwenClient, doc_index: DocumentIndex,
                 vector_indexer: VectorIndexer = None,
                 token_budget: int = 5_000_000, model: str = "qwen-plus"):
        self.qwen = qwen
        self.doc_index = doc_index
        self.vector = vector_indexer
        self.token_budget = token_budget
        self.model = model
        self.memory = AgentMemory(doc_index)
        self.cot_trails = []

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(
            self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        is_full_doc = total_doc_chars <= self.FULL_DOC_THRESHOLD

        # Card匹配（零Token）
        card_hints = self.memory.get_card_match_hints(q_text, options, doc_ids)

        # Step 1: 证据收集
        if is_full_doc:
            evidence_text = ""
            for doc_id in doc_ids:
                ft = self.doc_index.get_doc_full_text(doc_id)
                if ft:
                    compressed = compress_whitespace(ft)
                    evidence_text += f"\n=== 文档 {doc_id} (全文) ===\n{compressed}\n"
        else:
            # 长文档：逐选项检索证据 → 合并
            evidence_text = self._retrieve_all_options_evidence(
                q_text, options, doc_ids, card_hints, answer_format)

        # Step 2: 1次统一CoT验证
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")
        prompt = UNIFIED_VERIFY_PROMPT.format(
            evidence=evidence_text,
            question=q_text,
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
            option_a=options.get("A", ""),
            option_b=options.get("B", ""),
            option_c=options.get("C", ""),
            option_d=options.get("D", ""),
            answer_hint={
                "mcq": "一个大写字母(A/B/C/D)",
                "tf": "A或B",
                "multi": "多个大写字母按字母序(如ABC)，只包含有明确原文支持的选项",
            }.get(answer_format, ""),
        )

        try:
            result = self.qwen.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, max_tokens=3000, timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""

        answer = extract_answer_from_response(raw_response, answer_format)

        # Step 3: Self-Critique（多选题）
        if answer_format == "multi" and answer and len(answer) >= 2:
            critique_prompt = (
                f"多选题 {answer} 的二次校验。\n"
                f"对每个已选选项，确认文档证据中是否有明确原文支持。\n"
                f"无明确支持则删除。不能添加新选项。\n"
                f"最终答案："
            )
            try:
                critique_result = self.qwen.chat(
                    [{"role": "user", "content": critique_prompt}],
                    temperature=0.0, max_tokens=256, timeout=60,
                )
                corrected = extract_answer_from_response(critique_result["content"], "multi")
                if corrected and set(corrected).issubset(set(answer)):
                    answer = corrected
            except:
                pass

        answer = self._post_process(answer, answer_format)
        self.memory.questions_answered += 1

        self.cot_trails.append({
            "qid": qid, "domain": domain,
            "answer": answer, "answer_format": answer_format,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": is_full_doc,
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
        }

    def _retrieve_all_options_evidence(self, q_text: str, options: dict,
                                        doc_ids: list, card_hints: dict,
                                        answer_format: str) -> str:
        """为所有选项分别检索证据，去重合并"""
        per_option_limit = self.EVIDENCE_PER_OPTION.get(answer_format, 6000)
        total_limit = self.TOTAL_EVIDENCE_LIMIT.get(answer_format, 25000)

        # 全局检索（问题+所有选项）
        full_query = q_text + " " + " ".join(options.values())
        bm25_results = self.doc_index.search_bm25(full_query, top_k=25, doc_ids=doc_ids)

        # 逐选项定向检索
        option_evidence = {}
        seen_chunk_texts = set()

        for opt_label in sorted(options.keys()):
            opt_text = options.get(opt_label, "")
            if not opt_text:
                continue

            opt_query = f"{q_text} {opt_text}"
            opt_evidence_parts = []

            # 1. BM25检索（选项定向）
            opt_bm25 = self.doc_index.search_bm25(opt_query, top_k=10, doc_ids=doc_ids)
            for chunk in opt_bm25:
                text = chunk.get("text", "")
                text_key = text[:100]
                if text_key not in seen_chunk_texts:
                    seen_chunk_texts.add(text_key)
                    opt_evidence_parts.append(text)

            # 2. 向量检索（选项定向）
            if self.vector:
                vec_results = self.vector.search_vector(opt_query, top_k=10, doc_ids=doc_ids)
                for idx, score in vec_results:
                    if idx < len(self.doc_index.chunks):
                        chunk = self.doc_index.chunks[idx]
                        text = chunk.get("text", "")
                        text_key = text[:100]
                        if text_key not in seen_chunk_texts:
                            seen_chunk_texts.add(text_key)
                            opt_evidence_parts.append(text)

            # 3. 关键词检索
            opt_keywords = extract_query_keywords(opt_text)
            all_keywords = list(set(opt_keywords +
                                   card_hints.get("entity_terms", []) +
                                   card_hints.get("key_term_hits", [])))
            all_doc_chunks = self.doc_index.get_chunks_by_doc_ids(doc_ids)
            kw_scored = []
            for chunk in all_doc_chunks:
                text = chunk.get("text", "")
                score = keyword_match_score(all_keywords, text)
                if score > 0:
                    text_key = text[:100]
                    if text_key not in seen_chunk_texts:
                        kw_scored.append((score, text, text_key))
            kw_scored.sort(key=lambda x: -x[0])
            for score, text, text_key in kw_scored[:10]:
                if text_key not in seen_chunk_texts:
                    seen_chunk_texts.add(text_key)
                    opt_evidence_parts.append(text)

            # 4. 条款精准定位
            for doc_id in doc_ids:
                for clause_ref in card_hints.get("clause_refs", []):
                    clause_text = self.memory.locate_clause_text(doc_id, clause_ref)
                    if clause_text and clause_text[:100] not in seen_chunk_texts:
                        seen_chunk_texts.add(clause_text[:100])
                        opt_evidence_parts.append(clause_text)
                # 选项文本中的条款引用
                for m in CLAUSE_PATTERN.finditer(opt_text):
                    ref = m.group()
                    clause_text = self.memory.locate_clause_text(doc_id, ref)
                    if clause_text and clause_text[:100] not in seen_chunk_texts:
                        seen_chunk_texts.add(clause_text[:100])
                        opt_evidence_parts.append(clause_text)

            # 截断到每选项限制
            combined = "\n".join(opt_evidence_parts)
            option_evidence[opt_label] = combined[:per_option_limit]

        # 合并所有选项证据
        total_evidence = ""
        for opt_label in sorted(option_evidence.keys()):
            total_evidence += f"\n=== 选项{opt_label}相关证据 ===\n{option_evidence[opt_label]}\n"

        if len(total_evidence) > total_limit:
            total_evidence = total_evidence[:total_limit] + "\n[...证据已截断...]"

        return total_evidence

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
