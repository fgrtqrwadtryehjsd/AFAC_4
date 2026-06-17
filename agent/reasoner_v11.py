"""V11: Comprehensive Verification Agent — 终极版

=== 历史版本问题分析 ===
V10:  4次独立验证 → A=26 (好) 但4.66M Token ❌
V10.1: 统一验证 → 1.90M Token ✅ 但A=59 (label bias回归) ❌
V10.2: 选项洗牌 → A=59 (洗牌无效,证明是label bias而非order bias) ❌

=== 核心发现 ===
Position bias的本质是"标签偏见"(label bias):
- 模型偏好"A"标签本身，不管A选项在什么位置出现
- 简单的选项洗牌无效，因为标签A还是标签A
- V10的独立验证有效，因为每次只看1个选项，没有标签比较

=== V11六大改进 ===
1. ①②③④重标签: 随机将ABCD映射为①②③④，消除标签偏见
2. 强制评分(1-5): 先给每个选项独立评分，再基于评分选择，强制独立判断
3. 反偏见指令: 显式告知评分不受标签或位置影响
4. 全文阈值提升: 80K→100K，更多文档获得全文输入
5. Self-Critique增强: 所有题型做二次校验（不只多选）
6. 证据检索优化: 条款精准定位+实体聚焦+反证搜索

=== Token预算控制 ===
1次主验证 + 1次Self-Critique ≈ 22K tokens/question ≈ 2.2M total
TokenScore ≈ 0.56 (仍高于V10的0.068)
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
from agent.vector_indexer import VectorIndexer
from agent.postprocessor import extract_answer_from_response

# ============ 工具函数 ============

CLAUSE_PATTERN = re.compile(r'第[一二三四五六七八九十百千万零〇\d]+[条章节款项]', re.UNICODE)
MONEY_PATTERN = re.compile(r'[\d,.]+[万亿]?[元美元人民币]')
PERCENT_PATTERN = re.compile(r'[\d.]+%|[\d.]+百分之')
DATE_PATTERN = re.compile(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}')
ENTITY_PATTERN = re.compile(r'[\u4e00-\u9fff]{2,20}(?:股份|有限|集团|公司|银行|保险|证券|基金|信托)')

# ①②③④ 标签集
NEUTRAL_LABELS = ['①', '②', '③', '④']


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


# ============ 领域Prompt（增强版）============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": """你是保险条款审查员。严格依据文档原文逐选项判断。

关键区分：
- 身故保险金 ≠ 已交保费 ≠ 现金价值 ≠ 账户价值
- 退保金额 = 现金价值 - 退保费用
- 等待期内出险 ≠ 不承担保险责任（可能仅退还保费）
- 犹豫期退保 ≠ 逾期退保（扣除方式不同）

判断规则：
- 只有原文"明确、直接、完整"支持才判定✅
- 原文只提部分要素 → ⚠️部分支持
- 原文未提及或与选项矛盾 → ❌无支持
- 不能推断、联想或补充原文未说的内容""",

    "regulatory": """你是金融法规审查员。严格依据法规原文逐选项判断。

关键区分：
- "应当"/"必须"/"不得" = 强制性规定
- "可以" = 授权性规定（可以做，也可以不做）
- "经股东大会审议" ≠ "经特别决议通过"
- "经董事会决议" ≠ "经股东大会决议"
- "公开披露" ≠ "向监管报告"

判断规则：
- 只有原文"明确、直接、完整"支持才判定✅
- "可以"不能当作"必须"的证据
- 不能推断、联想或补充原文未说的内容""",

    "financial_contracts": """你是金融合同审查员。严格依据合同条款原文逐选项判断。

关键区分：
- 主体评级 ≠ 债项评级（评级对象不同）
- 违约事件须满足合同明确定义（不能类推）
- "经持有人会议同意" ≠ "自动生效"
- "加速到期" ≠ "立即到期"（有不同触发条件）

判断规则：
- 只有原文"明确、直接、完整"支持才判定✅
- 合同条款中的定义优先于一般理解
- 不能推断、联想或补充原文未说的内容""",

    "financial_reports": """你是财务报表审查员。严格依据年报精确数字逐选项判断。

关键区分：
- 同比 = 与去年同期比较
- 环比 = 与上一报告期比较
- "拟派发" ≠ "已派发"（仅是预案）
- 合并报表 ≠ 母公司报表
- 归母净利润 ≠ 净利润
- "审议通过" ≠ "实施完成"

判断规则：
- 只有原文精确数字"明确、直接"支持才判定✅
- 金额单位注意转换（万元 vs 元 vs 亿元）
- 不能用估算值代替报告值""",

    "research": """你是行业研报审查员。严格依据研报数据逐选项判断。

关键区分：
- "预计/预期/预测" ≠ "实际/已实现"
- "目标价" ≠ "当前价"
- "同比增速" ≠ "环比增速"
- "行业数据" ≠ "公司数据"

判断规则：
- 只有原文数据"明确、直接"支持才判定✅
- 预测数据必须与选项时间范围匹配
- 不能用旧数据替代新数据""",
}


# ============ V11核心Prompt: ①②③④重标签 + 强制评分 ============

V11_VERIFY_PROMPT = """## 任务
你是证据审查员。严格依据文档原文，对每个选项独立评分和判断。
注意：选项已用①②③④随机编号，编号与正确与否完全无关。

## 文档证据
{evidence}

## 问题
{question}

## 选项（随机编号，编号不暗示正确性）
{shuffled_options}

## 逐选项评分规则
对每个选项，先搜索原文相关语句，然后评分：
- 5分 = 原文有直接、完整、精确的支持语句
- 4分 = 原文有较明确的支持，但表述有细微差异
- 3分 = 原文有部分相关，但不足以完全支持
- 2分 = 原文只有间接或模糊的相关性
- 1分 = 原文无相关内容，或与选项矛盾

重要：评分只看原文证据，不受选项编号、位置或任何先验判断影响。

{option_scoring_sections}

## 最终答案
根据以上评分，选择评分≥4的选项。
{answer_format_hint}

请用选项编号(①②③④)作答。"""


# ============ Self-Critique Prompt（增强版，所有题型）============

CRITIQUE_PROMPT_MCQ = """对单选题 {answer} 的二次校验：

1. 回顾原文中支持选项{answer}的具体语句
2. 回顾原文中是否有更匹配的其他选项
3. 如果发现其他选项有更强的原文支持，改为该选项
4. 如果确认{answer}有最强原文支持，保持不变

最终答案："""

CRITIQUE_PROMPT_TF = """对判断题 {answer} 的二次校验：

1. 选项A(正确)的原文支持：
2. 选项B(不正确)的原文支持：
3. 如果B的反驳证据更强，改选B；否则保持A
4. 如果两个方向都没有强证据，保持当前选择

最终答案："""

CRITIQUE_PROMPT_MULTI = """对多选题 {answer} 的二次校验：

逐个检查已选选项：
{option_checks}

规则：
- 无明确原文支持(评分<4)则删除
- 不能添加新选项
- 只保留评分≥4的选项

最终答案："""


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
        hints = {"clause_refs": [], "entity_terms": [], "key_term_hits": [], "metric_contexts": []}
        full_query = question + " " + " ".join(options.values())
        for m in CLAUSE_PATTERN.finditer(full_query):
            hints["clause_refs"].append(m.group())
        # 数值型检索线索
        for m in MONEY_PATTERN.finditer(full_query):
            hints["metric_contexts"].append(m.group())
        for m in PERCENT_PATTERN.finditer(full_query):
            hints["metric_contexts"].append(m.group())
        for m in DATE_PATTERN.finditer(full_query):
            hints["metric_contexts"].append(m.group())
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

    def locate_clause_text(self, doc_id: str, clause_ref: str, max_len: int = 3000) -> str:
        full_text = self.doc_index.get_doc_full_text(doc_id)
        if not full_text:
            return ""
        pos = full_text.find(clause_ref)
        if pos < 0:
            return ""
        next_clause = CLAUSE_PATTERN.search(full_text, pos + len(clause_ref))
        end = next_clause.start() if next_clause else len(full_text)
        return full_text[pos:min(end, pos + max_len)].strip()

    def locate_metric_context(self, doc_id: str, metric_value: str, context_radius: int = 500) -> str:
        full_text = self.doc_index.get_doc_full_text(doc_id)
        if not full_text:
            return ""
        pos = full_text.find(metric_value)
        if pos < 0:
            return ""
        start = max(0, pos - context_radius)
        end = min(len(full_text), pos + len(metric_value) + context_radius)
        return full_text[start:end].strip()


# ============ V11 Verification Agent ============

class ReasoningAgentV11:
    """V11: Comprehensive Verification Agent

    六大改进：
    1. ①②③④重标签 → 消除ABCD标签偏见
    2. 强制评分(1-5) → 强制独立判断每个选项
    3. 反偏见指令 → 显式告知评分不受标签/位置影响
    4. 全文阈值100K → 更多文档全文输入
    5. Self-Critique增强 → 所有题型二次校验
    6. 证据检索优化 → 条款精准定位+数值上下文+实体聚焦
    """

    FULL_DOC_THRESHOLD = 100000  # 80K→100K

    EVIDENCE_PER_OPTION = {
        "mcq": 6000,
        "tf": 5000,
        "multi": 8000,
    }

    TOTAL_EVIDENCE_LIMIT = {
        "mcq": 20000,
        "tf": 10000,
        "multi": 35000,
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
            evidence_text = self._retrieve_all_options_evidence(
                q_text, options, doc_ids, card_hints, answer_format)

        # Step 2: ①②③④重标签映射
        original_labels = sorted(options.keys())
        neutral_labels = NEUTRAL_LABELS[:len(original_labels)]
        shuffled_originals = original_labels.copy()
        random.shuffle(shuffled_originals)
        # 映射: shuffled_originals[i] → neutral_labels[i]
        # 例如: C→①, A→②, D→③, B→④
        orig_to_neutral = dict(zip(shuffled_originals, neutral_labels))
        neutral_to_orig = {v: k for k, v in orig_to_neutral.items()}

        # 构建洗牌后的选项文本
        shuffled_options_text = "\n".join(
            f"选项{orig_to_neutral[lbl]}. {options[lbl]}" for lbl in shuffled_originals)

        # 构建评分段落
        scoring_sections = []
        for lbl in shuffled_originals:
            n_lbl = orig_to_neutral[lbl]
            opt_text = options.get(lbl, "")
            scoring_sections.append(
                f"**选项{n_lbl}**：{opt_text}\n"
                f"原文相关语句：\n"
                f"评分(1-5)：___\n"
                f"理由："
            )
        option_scoring_sections = "\n\n".join(scoring_sections)

        answer_format_hint = {
            "mcq": "单选：选评分最高的1个选项(①②③④)",
            "tf": "判断：选评分≥4的选项(①②③④)",
            "multi": "多选：选所有评分≥4的选项(①②③④)，按编号顺序排列",
        }.get(answer_format, "")

        # Step 3: 1次统一验证（重标签+强制评分）
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")
        prompt = V11_VERIFY_PROMPT.format(
            evidence=evidence_text,
            question=q_text,
            shuffled_options=shuffled_options_text,
            option_scoring_sections=option_scoring_sections,
            answer_format_hint=answer_format_hint,
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

        # 从①②③④映射回ABCD
        neutral_answer = self._extract_neutral_answer(raw_response, answer_format, len(original_labels))
        answer = self._map_neutral_to_original(neutral_answer, neutral_to_orig)

        # Step 4: Self-Critique（所有题型）
        if answer and answer_format == "mcq":
            critique = CRITIQUE_PROMPT_MCQ.format(answer=answer)
            try:
                critique_result = self.qwen.chat(
                    [{"role": "user", "content": critique}],
                    temperature=0.0, max_tokens=256, timeout=60,
                )
                corrected = extract_answer_from_response(critique_result["content"], "mcq")
                if corrected and corrected in "ABCD":
                    answer = corrected
            except:
                pass
        elif answer and answer_format == "tf":
            critique = CRITIQUE_PROMPT_TF.format(answer=answer)
            try:
                critique_result = self.qwen.chat(
                    [{"role": "user", "content": critique}],
                    temperature=0.0, max_tokens=256, timeout=60,
                )
                corrected = extract_answer_from_response(critique_result["content"], "tf")
                if corrected and corrected in "AB":
                    answer = corrected
            except:
                pass
        elif answer and answer_format == "multi" and len(answer) >= 2:
            option_checks = "\n".join(
                f"- 选项{c}: 评分≥4? 原文依据?"
                for c in sorted(answer)
            )
            critique = CRITIQUE_PROMPT_MULTI.format(answer=answer, option_checks=option_checks)
            try:
                critique_result = self.qwen.chat(
                    [{"role": "user", "content": critique}],
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
            "label_mapping": str(orig_to_neutral),
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
        }

    def _extract_neutral_answer(self, response: str, answer_format: str,
                                 num_options: int) -> str:
        """从模型回复中提取①②③④格式答案"""
        if not response:
            return ""

        # 查找①②③④格式的答案
        neutral_pattern = re.compile(r'[①②③④]+')
        matches = neutral_pattern.findall(response)

        if matches:
            # 取最后一个匹配（通常是最终答案）
            last_match = matches[-1]
            return last_match

        # 回退: 查找ABCD格式（模型可能忽略指令用ABCD）
        for c in reversed(response):
            if c in "ABCD":
                if answer_format == "mcq":
                    return c
                elif answer_format == "tf":
                    if c in "AB":
                        return c
                elif answer_format == "multi":
                    letters = sorted(set(ch for ch in response if ch in "ABCD"))
                    if len(letters) > 3:
                        letters = letters[:3]
                    return "".join(letters) if letters else ""

        # 兜底: 用编号格式
        num_pattern = re.compile(r'[1-4]+')
        num_matches = num_pattern.findall(response)
        if num_matches:
            last = num_matches[-1]
            num_to_neutral = {'1': '①', '2': '②', '3': '③', '4': '④'}
            return "".join(num_to_neutral.get(c, '') for c in last)

        return ""

    def _map_neutral_to_original(self, neutral_answer: str,
                                  neutral_to_orig: dict) -> str:
        """将①②③④映射回原始ABCD标签"""
        if not neutral_answer:
            return ""
        mapped = []
        for c in neutral_answer:
            if c in neutral_to_orig:
                mapped.append(neutral_to_orig[c])
            # 如果模型回退到ABCD，直接保留
            elif c in "ABCD":
                mapped.append(c)
        return "".join(sorted(set(mapped))) if mapped else ""

    def _retrieve_all_options_evidence(self, q_text: str, options: dict,
                                        doc_ids: list, card_hints: dict,
                                        answer_format: str) -> str:
        """为所有选项分别检索证据，去重合并"""
        per_option_limit = self.EVIDENCE_PER_OPTION.get(answer_format, 6000)
        total_limit = self.TOTAL_EVIDENCE_LIMIT.get(answer_format, 25000)

        full_query = q_text + " " + " ".join(options.values())

        option_evidence = {}
        seen_chunk_texts = set()

        for opt_label in sorted(options.keys()):
            opt_text = options.get(opt_label, "")
            if not opt_text:
                continue

            opt_query = f"{q_text} {opt_text}"
            opt_evidence_parts = []

            # 1. BM25检索
            opt_bm25 = self.doc_index.search_bm25(opt_query, top_k=10, doc_ids=doc_ids)
            for chunk in opt_bm25:
                text = chunk.get("text", "")
                text_key = text[:100]
                if text_key not in seen_chunk_texts:
                    seen_chunk_texts.add(text_key)
                    opt_evidence_parts.append(text)

            # 2. 向量检索
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

            # 4. 条款精准定位（扩大上下文到3000字）
            for doc_id in doc_ids:
                for clause_ref in card_hints.get("clause_refs", []):
                    clause_text = self.memory.locate_clause_text(doc_id, clause_ref, max_len=3000)
                    if clause_text and clause_text[:100] not in seen_chunk_texts:
                        seen_chunk_texts.add(clause_text[:100])
                        opt_evidence_parts.append(clause_text)
                # 选项文本中的条款引用
                for m in CLAUSE_PATTERN.finditer(opt_text):
                    ref = m.group()
                    clause_text = self.memory.locate_clause_text(doc_id, ref, max_len=3000)
                    if clause_text and clause_text[:100] not in seen_chunk_texts:
                        seen_chunk_texts.add(clause_text[:100])
                        opt_evidence_parts.append(clause_text)

            # 5. 数值型精准定位（V11新增）
            for doc_id in doc_ids:
                for metric_val in card_hints.get("metric_contexts", []):
                    metric_ctx = self.memory.locate_metric_context(doc_id, metric_val)
                    if metric_ctx and metric_ctx[:100] not in seen_chunk_texts:
                        seen_chunk_texts.add(metric_ctx[:100])
                        opt_evidence_parts.append(metric_ctx)

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
