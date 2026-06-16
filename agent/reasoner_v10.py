"""V10: Verification Agent — 逐选项验证后判决

核心范式转变：
  V9:  Question → Retrieval → Evidence → CoT → Answer (单阶段RAG)
  V10: Question → Option Decomposition → Per-Option Retrieval → Per-Option Verification → Answer Aggregation

关键创新：
1. 选项独立验证：对ABCD逐个检索证据+验证，消除position bias
2. 短文档(≤80K)全文输入，长文档问题驱动动态证据压缩
3. 证据审查员角色：模型从答案生成器变为证据审查员
4. 多选题必须完成ABCD全部验证后再聚合答案
5. 冲突/不足时触发二次检索验证循环
6. 证据投票+一致性校验生成最终答案
"""
import os
import json
import re
import time
from collections import defaultdict
from agent.config import RESULTS_DIR, MODEL_MAX_TOKENS
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.vector_indexer import VectorIndexer
from agent.postprocessor import extract_answer_from_response

# ============ 复用Card提取和关键词工具 ============

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
    """规则提取文档Card（零Token成本）"""
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
    """从文本中提取查询关键词"""
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
    """L1结构压缩：去除冗余空白，保留所有事实"""
    # 去除多余空行（3行以上变2行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除行内多余空格
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # 去除行尾空格
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


# ============ 领域System Prompt ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": "你是保险条款审查员。严格依据文档原文判断每个命题的真假。只有明确原文条款支持才判定为真。身故保险金≠已交保费≠现金价值≠账户价值，严格区分。退保金额=现金价值-退保费用。保险责任须同时满足所有条件。",

    "regulatory": "你是金融法规审查员。严格依据法规原文判断每个命题的真假。\"应当\"/\"必须\"/\"不得\"=强制性；\"可以\"=授权性。\"经股东大会审议\"≠\"经特别决议通过\"。修改公司章程=必须经特别决议通过。只有明确原文条款支持才判定为真。",

    "financial_contracts": "你是金融合同审查员。严格依据合同条款原文判断每个命题的真假。主体评级≠债项评级。违约事件须满足合同明确定义。只有明确原文条款支持才判定为真。",

    "financial_reports": "你是财务报表审查员。严格依据年报精确数字判断每个命题的真假。同比=去年同期；环比=上期。\"拟派发\"≠\"已派发\"。经营/投资/筹资现金流严格区分。只有明确原文数据支持才判定为真。",

    "research": "你是行业研报审查员。严格依据研报数据判断每个命题的真假。\"预计/预期\"≠\"实际\"。图表数据最权威。只有明确原文数据支持才判定为真。",
}


# ============ 选项验证Prompt ============

OPTION_VERIFY_PROMPT = """## 任务
你是证据审查员。判断选项{option_label}是否有明确的文档原文支持。

## 文档证据
{evidence}

## 问题
{question}

## 待验证选项
{option_label}. {option_text}

## 审查要求
1. 从文档证据中搜索与该选项直接相关的原文
2. 引用原文的精确语句（不能改写或概括）
3. 判断原文是否直接、完整地支持该选项

特别注意：
- "应当"≠"可以"，"召开前"≠"通知中"，精确用语必须匹配
- 部分支持≠完全支持，选项的所有要素都必须有原文支持
- 如果存在矛盾证据（原文中有相反规定），必须指出

## 输出格式
支持原文：「精确引用原文语句」
反驳原文：「如果存在矛盾，引用原文」（无则写"无"）
分析：基于原文精确用语的分析
置信度：高/中/低
结论：支持/不支持/部分支持"""

# 判断题专用验证Prompt
TF_VERIFY_PROMPT = """## 任务
你是证据审查员。判断以下命题是否有明确的文档原文支持。

## 文档证据
{evidence}

## 命题
{question}
{option_label}. {option_text}

## 审查要求
1. 从文档证据中搜索与该命题直接相关的原文
2. 引用原文的精确语句
3. 判断原文是否直接支持该命题为真

## 输出格式
支持原文：「精确引用原文语句」
反驳原文：「如果存在矛盾，引用原文」（无则写"无"）
分析：基于原文的分析
结论：支持(真)/不支持(假)"""

# 答案聚合Prompt
AGGREGATE_PROMPT = """## 任务
根据各选项的独立验证结果，生成最终答案。

## 问题
{question}

## 题型
{question_type}

## 各选项验证结果
{verification_results}

## 聚合规则
- 单选题(mcq)：选择唯一"支持"的选项
- 判断题(tf)：A=真(支持), B=假(不支持)
- 多选题(multi)：选择所有"支持"的选项，排除"不支持"和"部分支持但不完全"的

## 最终答案
{answer_hint}"""


# ============ Memory Agent ============

class AgentMemory:
    """文档记忆管理 — Card缓存 + 条款定位"""

    def __init__(self, doc_index: DocumentIndex):
        self.doc_index = doc_index
        self.cards = {}
        self.doc_access_count = defaultdict(int)
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
        card = self.get_card(doc_id)
        full_text = self.doc_index.get_doc_full_text(doc_id)
        if not full_text:
            return ""
        pos = full_text.find(clause_ref)
        if pos < 0:
            return ""
        next_clause = CLAUSE_PATTERN.search(full_text, pos + len(clause_ref))
        end = next_clause.start() if next_clause else len(full_text)
        return full_text[pos:min(end, pos + max_len)].strip()


# ============ V10 Verification Agent ============

class ReasoningAgentV10:
    """V10: 逐选项验证Agent (Option-by-Option Verification Agent)

    范式转变：
    - 模型角色：答案生成器 → 证据审查员
    - 推理方式：一次性判断所有选项 → 逐选项独立验证
    - 多选题：禁止一次性生成，必须完成ABCD全部验证后聚合
    - 短文档(≤80K)：全文输入
    - 长文档：问题驱动的动态证据压缩

    Token预算分配（100题，5M预算）：
    - 单选题(mcq): ~15K tokens/题 × ~25题 = 375K
    - 判断题(tf): ~10K tokens/题 × ~15题 = 150K
    - 多选题(multi): ~25K tokens/题 × ~60题 = 1.5M
    - 聚合: ~2K tokens/题 × 100题 = 200K
    - 预估总消耗: ~2.2M (TokenScore ≈ 0.56)
    - 剩余 ~2.8M 用于二次检索和验证循环
    """

    # 全文策略阈值
    FULL_DOC_THRESHOLD = 80000  # ≤80K的文档直接全文输入

    # 每选项证据量上限
    EVIDENCE_LIMITS = {
        "mcq": 8000,    # 单选题每选项8K证据
        "tf": 6000,     # 判断题每选项6K证据
        "multi": 10000, # 多选题每选项10K证据
    }

    # 每选项最大检索chunk数
    RETRIEVAL_TOP_K = {
        "mcq": 15,
        "tf": 10,
        "multi": 20,
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
        """主流程：选项分解 → 逐选项检索 → 逐选项验证 → 答案聚合"""
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        # ============ Step 1: 全文策略判断 ============
        total_doc_chars = sum(
            self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)

        is_full_doc = total_doc_chars <= self.FULL_DOC_THRESHOLD

        # ============ Step 2: Card匹配（零Token）============
        card_hints = self.memory.get_card_match_hints(q_text, options, doc_ids)

        # ============ Step 3: 全文准备 ============
        if is_full_doc:
            # 短文档：全文输入 + L1压缩
            full_evidence = ""
            for doc_id in doc_ids:
                ft = self.doc_index.get_doc_full_text(doc_id)
                if ft:
                    compressed = compress_whitespace(ft)
                    full_evidence += f"\n=== 文档 {doc_id} (全文) ===\n{compressed}\n"
        else:
            # 长文档：仅标记为需要逐选项检索
            full_evidence = None

        # ============ Step 4: 逐选项验证 ============
        option_labels = sorted(options.keys())  # A, B, C, D
        verification_results = {}

        for opt_label in option_labels:
            opt_text = options.get(opt_label, "")
            if not opt_text:
                continue

            # 4a. 为该选项构建检索query
            option_query = f"{q_text} {opt_text}"

            # 4b. 为该选项检索证据
            if is_full_doc:
                # 短文档：直接使用全文证据
                opt_evidence = full_evidence
            else:
                # 长文档：逐选项动态检索
                opt_evidence = self._retrieve_for_option(
                    q_text, opt_label, opt_text,
                    doc_ids, card_hints, answer_format)

            # 4c. 调用Qwen验证该选项
            verification = self._verify_option(
                q_text, opt_label, opt_text,
                opt_evidence, domain, answer_format)

            verification_results[opt_label] = verification

            # 4d. 检查是否需要二次检索
            if verification["conclusion"] == "部分支持" and not is_full_doc:
                # 证据不足，触发二次检索
                retry_evidence = self._retrieve_for_option_retry(
                    q_text, opt_label, opt_text,
                    doc_ids, card_hints, answer_format)
                retry_verification = self._verify_option(
                    q_text, opt_label, opt_text,
                    retry_evidence, domain, answer_format)
                # 用二次验证更新
                verification_results[opt_label] = retry_verification

        # ============ Step 5: 答案聚合 ============
        final_answer = self._aggregate_answers(
            q_text, options, verification_results, answer_format)

        # ============ Step 6: 一致性校验 ============
        final_answer = self._consistency_check(
            final_answer, verification_results, answer_format)

        # 后处理
        final_answer = self._post_process(final_answer, answer_format)

        # 记录CoT trail
        self.cot_trails.append({
            "qid": qid,
            "domain": domain,
            "answer": final_answer,
            "answer_format": answer_format,
            "evidence_chars": len(full_evidence) if full_evidence else 0,
            "total_doc_chars": total_doc_chars,
            "is_full_doc": is_full_doc,
            "verifications": verification_results,
        })

        return {
            "qid": qid,
            "answer": final_answer,
            "evidence_chars": len(full_evidence) if full_evidence else 0,
            "total_doc_chars": total_doc_chars,
        }

    def _retrieve_for_option(self, q_text: str, opt_label: str, opt_text: str,
                             doc_ids: list, card_hints: dict, answer_format: str) -> str:
        """为单个选项动态检索证据（长文档策略）"""

        evidence_limit = self.EVIDENCE_LIMITS.get(answer_format, 8000)
        top_k = self.RETRIEVAL_TOP_K.get(answer_format, 15)

        # 构建选项专属查询
        option_query = f"{q_text} {opt_text}"
        option_keywords = extract_query_keywords(opt_text)

        # 1. BM25检索（问题+选项文本）
        bm25_results = self.doc_index.search_bm25(option_query, top_k=top_k, doc_ids=doc_ids)

        # 2. 语义向量检索（选项定向）
        vector_chunks = []
        if self.vector:
            vec_results = self.vector.search_vector(option_query, top_k=top_k, doc_ids=doc_ids)
            for idx, score in vec_results:
                if idx < len(self.doc_index.chunks):
                    chunk = self.doc_index.chunks[idx]
                    if chunk.get("doc_id") in doc_ids:
                        vector_chunks.append(chunk)

        # 3. 关键词检索
        all_keywords = list(set(
            option_keywords +
            card_hints.get("entity_terms", []) +
            card_hints.get("key_term_hits", [])
        ))
        all_doc_chunks = self.doc_index.get_chunks_by_doc_ids(doc_ids)
        kw_scored = []
        for chunk in all_doc_chunks:
            text = chunk.get("text", "")
            score = keyword_match_score(all_keywords, text)
            if score > 0:
                kw_scored.append((score, chunk))
        kw_scored.sort(key=lambda x: -x[0])
        kw_results = [chunk for _, chunk in kw_scored[:top_k]]

        # 4. Card条款精准定位
        clause_evidence = []
        for doc_id in doc_ids:
            # 问题中提到的条款
            for clause_ref in card_hints.get("clause_refs", []):
                clause_text = self.memory.locate_clause_text(doc_id, clause_ref)
                if clause_text:
                    clause_evidence.append({
                        "doc_id": doc_id, "text": clause_text,
                        "clause_ref": clause_ref,
                    })
            # 选项文本中提到的条款
            for m in CLAUSE_PATTERN.finditer(opt_text):
                ref = m.group()
                clause_text = self.memory.locate_clause_text(doc_id, ref, max_len=3000)
                if clause_text:
                    clause_evidence.append({
                        "doc_id": doc_id, "text": clause_text,
                        "clause_ref": ref,
                    })

        # 5. RRF融合
        bm25_kw = self._rrf_fuse_chunks(bm25_results, kw_results, top_n=top_k)
        if self.vector:
            final_chunks = self._rrf_fuse_chunks_with_clause(
                clause_evidence, bm25_kw, vector_chunks, top_n=top_k * 2)
        else:
            # 无向量时：条款 + BM25/关键词
            seen_texts = set()
            final_chunks = list(clause_evidence)
            for chunk in bm25_kw:
                text_key = chunk.get("text", "")[:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    final_chunks.append(chunk)

        # 6. 组装证据文本
        evidence_text = ""
        for chunk in final_chunks:
            doc_id = chunk.get("doc_id", "")
            text = chunk.get("text", "")
            ref = chunk.get("clause_ref", "")
            label = f"文档{doc_id}" + (f" [{ref}]" if ref else "")
            evidence_text += f"\n--- {label} ---\n{text}\n"
            if len(evidence_text) > evidence_limit:
                break

        if len(evidence_text) > evidence_limit:
            evidence_text = evidence_text[:evidence_limit] + "\n[...证据已截断...]"

        return evidence_text

    def _retrieve_for_option_retry(self, q_text: str, opt_label: str, opt_text: str,
                                    doc_ids: list, card_hints: dict, answer_format: str) -> str:
        """二次检索：换一组检索词重新搜索"""
        evidence_limit = self.EVIDENCE_LIMITS.get(answer_format, 8000) + 4000  # 多给4K
        top_k = self.RETRIEVAL_TOP_K.get(answer_format, 15) + 5  # 多检索5个

        # 用更细的关键词
        retry_keywords = extract_query_keywords(opt_text)
        # 添加更多query变体
        retry_query = f"{opt_text}"

        bm25_results = self.doc_index.search_bm25(retry_query, top_k=top_k, doc_ids=doc_ids)

        # 向量检索用选项文本自身
        vector_chunks = []
        if self.vector:
            vec_results = self.vector.search_vector(retry_query, top_k=top_k, doc_ids=doc_ids)
            for idx, score in vec_results:
                if idx < len(self.doc_index.chunks):
                    chunk = self.doc_index.chunks[idx]
                    if chunk.get("doc_id") in doc_ids:
                        vector_chunks.append(chunk)

        # 关键词用选项文本的细粒度词
        kw_scored = []
        all_doc_chunks = self.doc_index.get_chunks_by_doc_ids(doc_ids)
        for chunk in all_doc_chunks:
            text = chunk.get("text", "")
            score = keyword_match_score(retry_keywords, text)
            if score > 0:
                kw_scored.append((score, chunk))
        kw_scored.sort(key=lambda x: -x[0])
        kw_results = [chunk for _, chunk in kw_scored[:top_k]]

        bm25_kw = self._rrf_fuse_chunks(bm25_results, kw_results, top_n=top_k)
        if self.vector:
            final_chunks = self._rrf_fuse_chunks_with_clause(
                [], bm25_kw, vector_chunks, top_n=top_k * 2)
        else:
            final_chunks = bm25_kw

        evidence_text = ""
        for chunk in final_chunks:
            text = chunk.get("text", "")
            evidence_text += f"\n---\n{text}\n"
            if len(evidence_text) > evidence_limit:
                break

        return evidence_text[:evidence_limit]

    def _verify_option(self, q_text: str, opt_label: str, opt_text: str,
                       evidence: str, domain: str, answer_format: str) -> dict:
        """对单个选项进行验证推理"""
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")

        if answer_format == "tf":
            prompt = TF_VERIFY_PROMPT.format(
                evidence=evidence,
                question=q_text,
                option_label=opt_label,
                option_text=opt_text,
            )
        else:
            prompt = OPTION_VERIFY_PROMPT.format(
                evidence=evidence,
                question=q_text,
                option_label=opt_label,
                option_text=opt_text,
            )

        try:
            result = self.qwen.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, max_tokens=1500, timeout=120,
            )
            response = result["content"]
        except Exception as e:
            print(f" [V_ERR:{e}]")
            return {"conclusion": "不支持", "confidence": "低", "raw_response": ""}

        # 解析验证结论
        conclusion = self._parse_verification_conclusion(response, answer_format)
        confidence = self._parse_confidence(response)

        return {
            "conclusion": conclusion,
            "confidence": confidence,
            "raw_response": response[:500],
        }

    def _parse_verification_conclusion(self, response: str, answer_format: str) -> str:
        """从验证结果中解析结论"""
        response_lower = response.lower()

        # 查找"结论"行
        conclusion_match = re.search(r'结论[：:]\s*(.*)', response)
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            if answer_format == "tf":
                if "支持" in conclusion_text and "不支持" not in conclusion_text:
                    return "支持"
                elif "不支持" in conclusion_text or "假" in conclusion_text:
                    return "不支持"
            else:
                if "不支持" in conclusion_text:
                    return "不支持"
                elif "部分支持" in conclusion_text:
                    return "部分支持"
                elif "支持" in conclusion_text:
                    return "支持"

        # 备用：检查全文倾向
        if answer_format == "tf":
            if "不支持" in response or "假" in response or "错误" in response:
                return "不支持"
            elif "支持" in response and "不支持" not in response:
                return "支持"
        else:
            if "不支持" in response and "支持" not in response.replace("不支持", ""):
                return "不支持"
            elif "部分支持" in response:
                return "部分支持"
            elif "支持" in response:
                return "支持"

        return "不支持"  # 默认不支持（宁缺毋滥）

    def _parse_confidence(self, response: str) -> str:
        """解析置信度"""
        conf_match = re.search(r'置信度[：:]\s*(高|中|低)', response)
        if conf_match:
            return conf_match.group(1)
        return "中"

    def _aggregate_answers(self, q_text: str, options: dict,
                           verification_results: dict, answer_format: str) -> str:
        """根据各选项验证结果聚合答案"""

        if answer_format == "mcq":
            # 单选题：选择唯一"支持"的选项
            supported = [label for label, result in verification_results.items()
                        if result["conclusion"] == "支持"]
            if len(supported) == 1:
                return supported[0]
            elif len(supported) > 1:
                # 多个支持，选置信度最高的
                high_conf = [s for s in supported if verification_results[s]["confidence"] == "高"]
                if high_conf:
                    return high_conf[0]
                return supported[0]  # fallback
            else:
                # 无明确支持，选"部分支持"中置信度最高的
                partial = [label for label, result in verification_results.items()
                          if result["conclusion"] == "部分支持"]
                if partial:
                    high_conf = [p for p in partial if verification_results[p]["confidence"] == "高"]
                    if high_conf:
                        return high_conf[0]
                    return partial[0]
                # 全部不支持 → 选第一个（兜底）
                return sorted(options.keys())[0]

        elif answer_format == "tf":
            # 判断题：A=真(支持), B=假(不支持)
            a_result = verification_results.get("A", {})
            if a_result.get("conclusion") == "支持":
                return "A"
            else:
                return "B"

        elif answer_format == "multi":
            # 多选题：选择所有"支持"的选项
            supported = [label for label, result in verification_results.items()
                        if result["conclusion"] == "支持"]
            if supported:
                return "".join(sorted(supported))

            # 无明确支持 → 选"部分支持"中置信度高的
            partial_high = [label for label, result in verification_results.items()
                           if result["conclusion"] == "部分支持"
                           and result["confidence"] == "高"]
            if partial_high:
                return "".join(sorted(partial_high))

            # 兜底：返回空（宁缺毋滥）
            return ""

        return ""

    def _consistency_check(self, answer: str, verification_results: dict,
                           answer_format: str) -> str:
        """一致性校验"""
        if not answer:
            return answer

        if answer_format == "multi":
            # 多选题：确保每个选中的选项都是"支持"
            selected = set(answer)
            confirmed = set()
            for opt in selected:
                result = verification_results.get(opt, {})
                if result.get("conclusion") == "支持":
                    confirmed.add(opt)
                elif result.get("conclusion") == "部分支持" and result.get("confidence") == "高":
                    confirmed.add(opt)
                # "不支持"的选项坚决排除

            result = "".join(sorted(confirmed))
            return result if result else answer  # 如果全部排除则保留原答案

        return answer

    def _rrf_fuse_chunks(self, bm25_results: list, kw_results: list,
                         bm25_weight: float = 1.0, kw_weight: float = 1.5,
                         top_n: int = 30) -> list:
        """RRF融合BM25和关键词检索结果"""
        rrf_scores = {}
        k = 60

        for rank, chunk in enumerate(bm25_results):
            text_key = chunk.get("text", "")[:100]
            rrf_scores[text_key] = rrf_scores.get(text_key, 0) + bm25_weight / (k + rank + 1)
            rrf_scores[f"__obj__{text_key}"] = chunk

        for rank, chunk in enumerate(kw_results):
            text_key = chunk.get("text", "")[:100]
            rrf_scores[text_key] = rrf_scores.get(text_key, 0) + kw_weight / (k + rank + 1)
            rrf_scores[f"__obj__{text_key}"] = chunk

        sorted_keys = sorted(
            [k for k in rrf_scores if not k.startswith("__obj__")],
            key=lambda x: -rrf_scores[x]
        )
        return [rrf_scores[f"__obj__{k}"] for k in sorted_keys[:top_n]
                if f"__obj__{k}" in rrf_scores]

    def _rrf_fuse_chunks_with_clause(self, clause_evidence: list,
                                      bm25_kw_results: list, vector_chunks: list,
                                      clause_weight: float = 2.0,
                                      bm25_kw_weight: float = 1.0,
                                      vector_weight: float = 1.2,
                                      top_n: int = 30) -> list:
        """三路RRF融合：条款 + BM25/关键词 + 向量"""
        rrf_scores = {}
        obj_map = {}
        k = 60

        for rank, item in enumerate(clause_evidence):
            text_key = item.get("text", "")[:100]
            rrf_scores[text_key] = rrf_scores.get(text_key, 0) + clause_weight / (k + rank + 1)
            obj_map[text_key] = item

        for rank, chunk in enumerate(bm25_kw_results):
            text_key = chunk.get("text", "")[:100]
            rrf_scores[text_key] = rrf_scores.get(text_key, 0) + bm25_kw_weight / (k + rank + 1)
            obj_map[text_key] = chunk

        for rank, chunk in enumerate(vector_chunks):
            text_key = chunk.get("text", "")[:100]
            rrf_scores[text_key] = rrf_scores.get(text_key, 0) + vector_weight / (k + rank + 1)
            obj_map[text_key] = chunk

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
        return [obj_map[k] for k in sorted_keys[:top_n] if k in obj_map]

    def _post_process(self, answer: str, answer_format: str) -> str:
        """答案后处理"""
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
        """保存CoT推理记录"""
        path = os.path.join(RESULTS_DIR, "eval_results_full.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)

        card_path = os.path.join(RESULTS_DIR, "document_cards.json")
        cards_data = {doc_id: card for doc_id, card in self.memory.cards.items()}
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(cards_data, f, ensure_ascii=False, indent=2)
