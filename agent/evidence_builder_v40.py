"""V40a 证据构建器 — 结构化事实包 + 倒U重排 (零 token)

## 用户三创新点落地
① 锚定增量状态机: 从 DocumentCard 提取结构化事实(数字+上下文)代替大段原文
② 上下文手术刀: 倒U型重排, 最重要事实放首尾, 对抗 LLM 中间迷失
③ 双Agent管家(简化版): 按重要性分级 — 题干命中科目高分保留原貌, 无关截断

## 核心资产
results/document_cards.json (68文档, 零token规则生成):
  - metrics: [{type: money/percent/date, value, context}] (BYD年报2854条)
  - clauses: [{ref, position, content_preview}]
  - entities / section_titles / key_terms

## 事实包构建逻辑
1. 从题干+选项提取"查询关键词" (财务科目/产品名/法规名/数字)
2. 在 doc 的 metrics.clauses 里用关键词匹配 context, 命中=高重要性
3. 按重要性排序 → 倒U重排 (top-N 放前, 次top-N 放后, 中间填次要)
4. 每选项预算分配 (SOP V40: multi 12-20K, mcq 8-12K, tf 4-6K)

## 零 token 保证
纯规则匹配 (字符串包含), 不调 LLM, 不调 embedding.
"""
import os
import json
import re
from collections import defaultdict
from agent.config import PROCESSED_DIR, RESULTS_DIR

CARD_PATH = os.path.join(RESULTS_DIR, "document_cards.json")

# 财务科目关键词 (财报/合同域核心查询词)
FIN_KEYWORDS = [
    "营业收入", "营收", "营业总收入", "归母净利润", "归属于上市公司股东的净利润",
    "净利润", "扣非净利润", "经营活动", "现金流量净额", "投资活动", "筹资活动",
    "研发投入", "研发费用", "毛利率", "净利率", "资产负债率", "每股收益",
    "现金分红", "利润分配", "派发", "股份回购", "回购", "净资产", "总资产",
    "同比增长", "同比下降", "同比", "环比",
]
# 合同/债券关键词
CONTRACT_KEYWORDS = [
    "转股价格", "向下修正", "赎回", "回售", "违约", "资产减值", "业绩补偿",
    "票面利率", "信用评级", "主体评级", "债项评级", "发行规模", "发行对象",
    "募集", "担保", "增信",
]
# 保险关键词
INS_KEYWORDS = [
    "身故保险金", "现金价值", "保险金额", "基本保险金额", "已交保费", "账户价值",
    "等待期", "犹豫期", "免责", "保单贷款", "退保", "年金", "分红",
    "受益人", "投保人", "被保险人",
]
# 法规关键词 (用题干书名号内容更准, 这里是补充)
REG_KEYWORDS = ["应当", "不得", "禁止", "可以", "工作日", "日", "个月", "年"]

DOMAIN_KEYWORDS = {
    "financial_reports": FIN_KEYWORDS,
    "financial_contracts": CONTRACT_KEYWORDS,
    "insurance": INS_KEYWORDS,
    "regulatory": REG_KEYWORDS,
    "research": FIN_KEYWORDS,  # 研报也有财务数据
}


class FactCardStore:
    """DocumentCard 加载与查询 (零 token)."""

    def __init__(self, card_path=CARD_PATH):
        self.cards = {}
        if os.path.exists(card_path):
            with open(card_path, encoding="utf-8") as f:
                self.cards = json.load(f)
        self._domain_map = self._build_domain_map()

    def _build_domain_map(self):
        """用 processed_data 目录补 doc_id → domain (card 无 domain 字段)."""
        m = {}
        for dom in os.listdir(PROCESSED_DIR):
            ddir = os.path.join(PROCESSED_DIR, dom)
            if not os.path.isdir(ddir):
                continue
            for root, dirs, files in os.walk(ddir):
                for f in files:
                    if f.endswith(".json") and f != "structured_index.json":
                        m[os.path.splitext(f)[0]] = dom
        return m

    def get_card(self, doc_id):
        return self.cards.get(doc_id)

    def domain_of(self, doc_id):
        return self._domain_map.get(doc_id, "")


def extract_query_keywords(question):
    """从题干+选项提取查询关键词.

    返回:
      keyword_set: set, 命中的域关键词
      book_titles: list, 书名号内内容 (法规名)
      numbers: list, 题干中的数字 (金额/百分比/年份)
    """
    q_text = question.get("question", "")
    options = question.get("options", {})
    full = q_text + " " + " ".join(str(v) for v in options.values())

    # 1. 域关键词
    domain = question.get("domain", "")
    domain_kws = DOMAIN_KEYWORDS.get(domain, [])
    keyword_set = {kw for kw in domain_kws if kw in full}

    # 2. 书名号 (法规名)
    book_titles = re.findall(r"《([^》]{2,40})》", full)

    # 3. 数字 (金额/百分比/年份, 用于 metric value 匹配)
    #    匹配 "5%""5.0%""803,964""542亿" 等
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:%|亿|万|元|亿元|万美元|个|天|日|月|年)?", full)
    numbers = [n.strip() for n in numbers if len(n.strip()) >= 2]

    return keyword_set, book_titles, numbers


def score_metric(metric, keywords, numbers):
    """给单条 metric 打重要性分 (创新点③ 简化版).

    - context 命中域关键词: +3/词 (高重要性, 保留原貌)
    - value 命中题干数字: +2 (精确数字命中)
    - money 类型基础分 +1 (金额比日期更有信息量)
    """
    score = 0
    ctx = metric.get("context", "")
    val = metric.get("value", "")
    mtype = metric.get("type", "")

    for kw in keywords:
        if kw in ctx:
            score += 3

    # 数字匹配: 去掉非数字字符后比较核心
    val_core = re.sub(r"[^\d]", "", val)
    for n in numbers:
        n_core = re.sub(r"[^\d]", "", n)
        if n_core and val_core and len(n_core) >= 3:
            if n_core in val_core or val_core in n_core:
                score += 2
                break

    if mtype == "money":
        score += 1
    elif mtype == "percent":
        score += 0.5

    return score


def build_fact_pack(question, doc_index, fact_store, max_chars=15000):
    """构建单题事实包 (创新点①②③).

    流程:
    1. 提取查询关键词
    2. 对每个 doc, 在 metrics/clauses 里匹配, 打重要性分
    3. 全局排序 → 倒U重排
    4. 按 doc 分组输出 (防跨文档错配)

    Returns:
        evidence: str, 事实包文本
        stats: dict, 构建统计 (命中数/各doc分配)
    """
    doc_ids = question.get("doc_ids", [])
    keywords, book_titles, numbers = extract_query_keywords(question)
    domain = question.get("domain", "")
    keywords = keywords | set(book_titles)

    all_facts = []  # [(score, doc_id, text)]
    for did in doc_ids:
        card = fact_store.get_card(did)
        full_text = doc_index.get_doc_full_text(did) or ""

        # 无 card → 回退 V20 方式 (头部 + 关键词定位)
        if not card:
            for kw in list(keywords)[:8]:
                pos = full_text.find(kw)
                if pos >= 0:
                    seg = full_text[pos:pos + 800].replace("\n", " ")
                    all_facts.append((3, did, f"[{kw}] {seg[:200]}"))
            all_facts.append((1, did, f"[文档头部] {full_text[:500].replace(chr(10),' ')}"))
            continue

        # === 财报域: metrics 为主 (数字型, 已验证 79% 覆盖) ===
        if domain == "financial_reports":
            metrics = card.get("metrics", [])
            scored = [(score_metric(m, keywords, numbers), m) for m in metrics]
            scored = [(s, m) for s, m in scored if s > 0]
            scored.sort(key=lambda x: -x[0])
            seen_ctx = set()
            for s, m in scored[:20]:
                ctx = m.get("context", "").strip().replace("\n", " ")
                ctx_key = ctx[:60]
                if ctx_key in seen_ctx:
                    continue
                seen_ctx.add(ctx_key)
                if len(ctx) > 200:
                    ctx = ctx[:200] + "..."
                all_facts.append((s, did, f"[{m.get('type','')}] {m.get('value','')}: {ctx}"))

        # === 保险域: key_terms + clauses 条款定位 (条款型, 非 metrics) ===
        elif domain == "insurance":
            key_terms = card.get("key_terms", [])
            matched_terms = []
            for kt in key_terms:
                term = kt.get("term", "") if isinstance(kt, dict) else str(kt)
                if term and any(term in k or k in term for k in keywords):
                    matched_terms.append(term)
                    pos = full_text.find(term)
                    if pos >= 0:
                        seg = full_text[pos:pos + 1200].replace("\n", " ")
                        all_facts.append((5, did, f"[术语:{term}] {seg[:300]}"))
            for cl in card.get("clauses", []):
                preview = cl.get("content_preview", "")
                ref = cl.get("ref", "")
                if any(kw in preview for kw in keywords) or any(kw in ref for kw in keywords):
                    all_facts.append((4, did, f"[条款 {ref}] {preview[:250]}"))
            if not matched_terms:
                for term in ["身故保险金", "现金价值", "保单贷款", "退保"]:
                    pos = full_text.find(term)
                    if pos >= 0:
                        seg = full_text[pos:pos + 1000].replace("\n", " ")
                        all_facts.append((2, did, f"[术语:{term}] {seg[:250]}"))

        # === 合同域: metrics + clauses 混合 ===
        elif domain == "financial_contracts":
            metrics = card.get("metrics", [])
            scored = [(score_metric(m, keywords, numbers), m) for m in metrics]
            scored = [(s, m) for s, m in scored if s > 0]
            scored.sort(key=lambda x: -x[0])
            for s, m in scored[:8]:
                ctx = m.get("context", "").strip().replace("\n", " ")[:200]
                all_facts.append((s, did, f"[{m.get('type','')}] {m.get('value','')}: {ctx}"))
            for cl in card.get("clauses", []):
                preview = cl.get("content_preview", "")
                if any(kw in preview for kw in keywords):
                    all_facts.append((4, did, f"[条款 {cl.get('ref','')}] {preview[:250]}"))

        # 其他域: 通用条款 + 头部
        else:
            for cl in card.get("clauses", []):
                preview = cl.get("content_preview", "")
                if any(kw in preview for kw in keywords):
                    all_facts.append((4, did, f"[条款 {cl.get('ref','')}] {preview[:250]}"))
            all_facts.append((1, did, f"[文档头部] {full_text[:800].replace(chr(10),' ')}"))

    if not all_facts:
        for did in doc_ids:
            full_text = doc_index.get_doc_full_text(did) or ""
            all_facts.append((0, did, f"[文档头部] {full_text[:2000].replace(chr(10),' ')}"))

    all_facts.sort(key=lambda x: -x[0])
    reordered = _u_shaped_reorder(all_facts)

    # 组装 (按出现顺序, 标注 doc_id)
    parts = []
    total = 0
    per_doc_count = defaultdict(int)
    for s, did, text in reordered:
        if total >= max_chars:
            break
        if total + len(text) > max_chars:
            text = text[: max_chars - total]
        parts.append(f"=== {did} ===\n{text}")
        per_doc_count[did] += 1
        total += len(text)

    evidence = "\n\n".join(parts)
    return evidence, {
        "n_facts": len(reordered),
        "n_keywords": len(keywords),
        "keywords": sorted(keywords)[:15],
        "per_doc": dict(per_doc_count),
        "evidence_chars": len(evidence),
    }


def _u_shaped_reorder(facts):
    """倒U型重排: 重要性最高放首尾, 最低放中间.

    输入已按 score 降序. 输出: [最高, 第3高, 第5高, ..., 最低, ..., 第4高, 第2高]
    即奇数位从头取, 偶数位从尾取, 使首尾都是高分项.
    """
    if len(facts) <= 2:
        return facts
    front = []  # 首部: 奇数索引 (0,2,4,...)
    back = []   # 尾部: 偶数索引 (1,3,5,...)
    for i, f in enumerate(facts):
        if i % 2 == 0:
            front.append(f)
        else:
            back.append(f)
    back.reverse()  # 尾部从低到高 → 拼接后尾部是高分
    return front + back
