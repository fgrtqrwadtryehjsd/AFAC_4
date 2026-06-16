"""V9: 语义向量融合检索 Agent (Semantic-Vector Fusion Retrieval Agent)

核心技术创新：
1. 语义向量检索（DashScope text-embedding-v3，不计入比赛Token！）
2. BM25 + Vector + Keyword 三路RRF融合
3. 选项定向检索（每个选项独立搜证据）
4. Card引导精准定位条款
5. 继承V7.1的全最优设计（无shuffle）

与V7.1的核心区别：
- V7.1: BM25 + 关键词（两路检索，召回有限）
- V9: BM25 + 向量 + 关键词（三路检索，大幅提升召回）

关键洞察：
- BM25精确但召回差（只匹配关键词）
- 向量搜索模糊但召回好（语义匹配）
- RRF融合取长补短
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


# ============ 复用V7.1的Card提取和关键词检索 ============

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
    
    term_candidates = defaultdict(int)
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
    for term in financial_terms:
        count = full_text.count(term)
        if count > 0:
            term_candidates[term] = count
    
    card["key_terms"] = [
        {"term": t, "count": c} for t, c in
        sorted(term_candidates.items(), key=lambda x: -x[1])[:30]
    ]
    
    return card


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
                context = metric.get("context", "")
                if value in full_query or any(kw in context for kw in full_query.split() if len(kw) >= 3):
                    hints["metric_terms"].append(value)
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


# ============ 领域 Prompt ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": """你是保险条款分析专家。严格依据文档原文判断每个选项。
- 身故保险金 ≠ 已交保费 ≠ 现金价值 ≠ 账户价值，严格区分
- 退保金额 = 现金价值 - 退保费用
- 保险责任须同时满足所有条件
- 只有明确原文条款支持才选""",

    "regulatory": """你是金融监管合规专家。严格依据法规原文判断每个选项。
- "应当"/"必须"/"不得" = 强制性；"可以" = 授权性
- "经股东大会审议" ≠ "经特别决议通过"
- 修改公司章程 = 必须经特别决议通过
- 只有明确原文条款支持才选""",

    "financial_contracts": """你是金融合同分析专家。严格依据合同条款原文判断每个选项。
- 主体评级 ≠ 债项评级
- 违约事件须满足合同明确定义
- 只有明确原文条款支持才选""",

    "financial_reports": """你是财务报表分析专家。严格依据年报精确数字判断每个选项。
- 同比 = 去年同期；环比 = 上期
- "拟派发" ≠ "已派发"
- 经营/投资/筹资现金流严格区分
- 只有明确原文数据支持才选""",

    "research": """你是行业研报分析专家。严格依据研报数据判断每个选项。
- "预计/预期" ≠ "实际"
- 图表数据最权威
- 只有明确原文数据支持才选""",
}


COT_PROMPT = """## 任务
严格依据文档证据，逐选项判断是否正确。

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 逐选项严格验证

对每个选项，你必须：
1. 先从文档中搜索与该选项直接相关的原文
2. 引用原文的精确语句（不能改写或概括）
3. 基于原文精确用语判断是否支持

**选项A**：{option_a}
搜索关键词：
原文精确引用：
分析：原文是否直接支持选项A的完整表述？
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

**选项B**：{option_b}
搜索关键词：
原文精确引用：
分析：原文是否直接支持选项B的完整表述？注意区分"应当"vs"可以"、"召开前"vs"通知中"等
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

**选项C**：{option_c}
搜索关键词：
原文精确引用：
分析：原文是否直接支持选项C的完整表述？
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

**选项D**：{option_d}
搜索关键词：
原文精确引用：
分析：原文是否直接支持选项D的完整表述？
判断：✅ 明确支持 / ❌ 无明确支持 / ⚠️ 部分支持但不完全

最终答案：{answer_hint}"""


SELF_CRITIQUE_PROMPT = """多选题 {answer} 的二次校验。
对每个已选选项，确认文档中是否有明确原文支持。无明确支持则删除。不能添加新选项。
最终答案："""


# ============ V9 Agent ============

class ReasoningAgentV9:
    """V9: 语义向量融合检索 Agent
    
    技术创新：
    1. DashScope text-embedding-v3 语义检索（不计入比赛Token！）
    2. BM25 + Vector + Keyword 三路RRF融合
    3. 选项定向向量检索（每个选项独立搜证据）
    4. 继承V7.1的Card系统和精确推理Prompt
    """
    
    def __init__(self, qwen: QwenClient, doc_index: DocumentIndex,
                 vector_indexer: VectorIndexer = None,
                 token_budget: int = 5_000_000, model: str = "qwen-plus"):
        self.qwen = qwen
        self.doc_index = doc_index
        self.vector = vector_indexer
        self.token_budget = token_budget
        self.memory = AgentMemory(doc_index)
        self.cot_trails = []
    
    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])
        
        # === Phase 1: Card匹配（零Token成本）===
        card_hints = self.memory.get_card_match_hints(q_text, options, doc_ids)
        
        # === Phase 2: 三路检索 ===
        
        # 2a. BM25 主查询
        bm25_results = self.doc_index.search_bm25(q_text, top_k=20, doc_ids=doc_ids)
        
        # 2b. 语义向量检索（核心创新！）
        # 每题只需1-2次Embedding API调用（不计入比赛Token）
        vector_results = []
        if self.vector:
            # 问题向量检索
            q_vector = self.vector.search_vector(q_text, top_k=15, doc_ids=doc_ids)
            
            # 合并问题+所有选项的向量检索（1次API调用）
            opt_queries = [f"{q_text} {options[k]}" for k in sorted(options.keys())]
            opt_vector = self.vector.search_vector_multi_query(
                opt_queries, top_k=15, doc_ids=doc_ids)
            
            # RRF融合向量搜索结果
            vector_results = self.vector.rrf_fuse(q_vector, opt_vector, top_n=25)
        
        # 2c. Card引导的关键词检索
        query_keywords = extract_query_keywords(q_text, options)
        search_keywords = list(set(
            query_keywords +
            card_hints["entity_terms"] +
            card_hints["key_term_hits"]
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
        
        # 2d. Card条款精准定位
        clause_evidence = []
        for doc_id in doc_ids:
            for clause_ref in card_hints["clause_refs"]:
                clause_text = self.memory.locate_clause_text(doc_id, clause_ref)
                if clause_text:
                    clause_evidence.append({
                        "doc_id": doc_id, "text": clause_text,
                        "chunk_type": "clause_precise", "clause_ref": clause_ref,
                    })
        
        # === Phase 3: 证据组装（三路RRF融合）===
        total_doc_chars = sum(
            self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        
        if total_doc_chars <= 50000:
            # 短文档：直接全文
            evidence_text = ""
            for doc_id in doc_ids:
                ft = self.doc_index.get_doc_full_text(doc_id)
                if ft:
                    evidence_text += f"\n=== 文档 {doc_id} (全文) ===\n{ft}\n"
        else:
            # 长文档：三路融合
            # BM25 + 关键词 → chunk级别融合
            bm25_kw = self._rrf_fuse_chunks(
                bm25_results, kw_results,
                bm25_weight=1.0, kw_weight=1.5, top_n=40)
            
            # 向量检索结果 → chunk级别
            vector_chunks = []
            if self.vector:
                for idx, score in vector_results:
                    if idx < len(self.doc_index.chunks):
                        chunk = self.doc_index.chunks[idx]
                        if chunk.get("doc_id") in doc_ids:
                            vector_chunks.append(chunk)
            
            # 三路最终融合
            final_chunks = self._rrf_fuse_chunks_3way(
                clause_evidence, bm25_kw, vector_chunks, top_n=50)
            
            # 证据量控制（与V7.1一致）
            if answer_format == "multi":
                max_evidence = 25000
            elif answer_format == "tf":
                max_evidence = 15000
            else:
                max_evidence = 20000
            
            evidence_text = ""
            for chunk in final_chunks:
                doc_id = chunk.get("doc_id", "")
                text = chunk.get("text", "")
                ref = chunk.get("clause_ref", "")
                label = f"文档 {doc_id}" + (f" [{ref}]" if ref else "")
                evidence_text += f"\n--- 来自{label} ---\n{text}\n"
                if len(evidence_text) > max_evidence:
                    break
            
            if len(evidence_text) > max_evidence:
                evidence_text = evidence_text[:max_evidence] + "\n[...证据已截断...]"
        
        # === Phase 4: CoT推理 ===
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")
        
        reasoning_prompt = COT_PROMPT.format(
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
                    {"role": "user", "content": reasoning_prompt},
                ],
                temperature=0.1, max_tokens=4096, timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""
        
        answer = extract_answer_from_response(raw_response, answer_format)
        
        # === Phase 5: Self-Critique（多选题）===
        if answer_format == "multi" and answer and len(answer) >= 2:
            try:
                critique_result = self.qwen.chat(
                    [{"role": "user", "content": SELF_CRITIQUE_PROMPT.format(answer=answer)}],
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
            "qid": qid,
            "domain": domain,
            "answer": answer,
            "clause_refs_found": len(card_hints["clause_refs"]),
            "card_entity_hits": len(card_hints["entity_terms"]),
            "card_key_term_hits": len(card_hints["key_term_hits"]),
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
            "card_cache_total": sum(self.memory.doc_access_count.values()),
        })
        
        return {"qid": qid, "answer": answer, "raw_response": raw_response}
    
    def _rrf_fuse_chunks(self, bm25_results, kw_results,
                         bm25_weight=1.0, kw_weight=1.5, top_n=40) -> list:
        """BM25 + 关键词 RRF融合 → 返回chunk列表"""
        chunk_scores = {}
        k = 60
        
        for rank, chunk in enumerate(bm25_results):
            key = chunk.get("text", "")[:100]
            rrf = bm25_weight / (k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf
        
        for rank, chunk in enumerate(kw_results):
            key = chunk.get("text", "")[:100]
            rrf = kw_weight / (k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf
        
        sorted_results = sorted(chunk_scores.values(), key=lambda x: -x["score"])
        return [item["chunk"] for item in sorted_results[:top_n]]
    
    def _rrf_fuse_chunks_3way(self, clause_results, bm25_kw_results,
                              vector_chunks, top_n=50) -> list:
        """三路RRF融合：条款精准 + BM25+关键词 + 向量"""
        chunk_scores = {}
        k = 60
        
        # 条款精准证据（权重最高）
        for rank, chunk in enumerate(clause_results):
            key = chunk.get("text", "")[:100]
            rrf = 2.0 / (k + rank + 1)  # 权重2.0
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf
        
        # BM25+关键词
        for rank, chunk in enumerate(bm25_kw_results):
            key = chunk.get("text", "")[:100]
            rrf = 1.0 / (k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf
        
        # 语义向量
        for rank, chunk in enumerate(vector_chunks):
            key = chunk.get("text", "")[:100]
            rrf = 1.2 / (k + rank + 1)  # 向量权重略高
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf
        
        sorted_results = sorted(chunk_scores.values(), key=lambda x: -x["score"])
        return [item["chunk"] for item in sorted_results[:top_n]]
    
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
