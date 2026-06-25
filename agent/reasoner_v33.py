"""V33: 精准单窗口检索

核心洞察（来自 fc_a_001 分析）:
- 债券说明书答案在文档前 200 字（封面）
- 年报财务数据在第 11 页（char ~10K）的摘要表
- 保险条款答案在特定条款号处
- 99% 的证据用不上，只需要精准的 600-800 字窗口

V33 架构（multi 题）:
  Step 0 (零LLM): 为每个选项定位精准文档 + 精准位置
    - insurance: 选项中提取产品名 → 匹配文档 → 问题关键词 str.find
    - financial_contracts: 文档前 1000 字 (封面含所有关键字段)
    - financial_reports: 找「主要会计数据」或「财务指标」段落 (~char 8K-15K)
    - regulatory: 选项数字/条款名 str.find
    - research: 问题引号内词 + 选项数字 str.find
  Step 1 (零LLM): 每个选项取 700 字上下文 (~2-4K 总证据)
  Step 2 (1次LLM, enable_thinking=False): 基于精炼证据答题

tf/mcq: 保留 V31 精炼路径 (V30)

Token 估算:
  multi 65题 × 3K × ~3tokens/char = ~585K tokens
  tf/mcq 35题 × 10K = ~350K tokens
  合计 ~935K tokens, TS = (5M-935K)/5M = 0.81
  Score(Acc=60%) = 60 × 0.943 = 56.6
  Score(Acc=70%) = 70 × 0.943 = 66.0
"""
import re
import json
import os
from agent.indexer import DocumentIndex
from agent.reasoner_v31 import ReasoningAgentV31
from agent.reasoner_v20 import DOMAIN_SYSTEM
from agent.postprocessor import extract_answer_from_response
from agent.config import RESULTS_DIR


# ============ 文档产品名索引（保险domain专用）============

# doc_id -> 产品名关键词列表（用于从选项文本匹配对应文档）
INS_DOC_PRODUCTS = {
    '1':  ['平安智盈金生', '智盈金生'],
    '2':  ['国寿增益宝', '增益宝'],
    '3':  ['众安白血病', '急性白血病'],
    '4':  ['平安安佑福', '安佑福'],
    '5':  ['平安e生保', 'e生保'],
    '6':  ['太保团体百万', '太平洋健康'],
    '7':  ['平安预防接种', '预防接种意外'],
    '8':  ['众安营运交通', '营运交通'],
    '9':  ['平安特种车', '特种车商业保险'],
    '10': ['众安特种车', '众安特种车商业'],
    '11': ['平安家财险', '家庭财产保险'],
    '13': ['众安食责险', '食品安全责任保险'],
    '14': ['平安食责险', '平安产险食品安全'],
    '16': ['平安富鸿金生', '富鸿金生'],
}


# ============ 精准定位函数 ============

SKIP_INS = 8000  # 保险文档前8K是目录

def _find_context(text: str, keywords: list, ctx_chars: int = 700,
                  skip: int = 0, before: int = 200) -> str:
    """在 text[skip:] 中搜索关键词列表，返回第一个命中的上下文窗口"""
    search_text = text[skip:]
    for kw in keywords:
        if not kw:
            continue
        pos = search_text.find(kw)
        if pos >= 0:
            abs_pos = skip + pos
            start = max(0, abs_pos - before)
            return text[start: start + ctx_chars]
    return ""


def _extract_quoted(text: str) -> list:
    """提取引号内的词"""
    return re.findall(r'["""«»「」『』【】《》](.*?)["""»「」』】》]', text)


def _extract_numbers(text: str) -> list:
    """提取 数字+单位 短语，e.g. '30个工作日', '6个月', '80%'"""
    # 去除空格再匹配（原文可能有 '30 个工作日'）
    clean = re.sub(r'(\d)\s+([一-鿿%])', r'\1\2', text)
    return re.findall(
        r'\d+(?:\.\d+)?(?:%|亿|万|元|年|个?月|个?工作日|天|条|款|倍)',
        clean)


SYNONYMS_MAP = {
    "保单贷款": ["借款", "保单借款"],
    "借款": ["保单贷款", "保单借款"],
    "犹豫期": ["犹豫撤单", "15天", "10天"],
    "营业收入": ["营收", "营业总收入"],
    "归母净利润": ["归属于上市公司股东的净利润"],
    "研发投入": ["研发费用", "研究开发费用"],
    "施行日期": ["施行", "生效日期"],
}

def _expand_kws(kws: list) -> list:
    """扩展同义词"""
    expanded = list(kws)
    for kw in kws:
        expanded.extend(SYNONYMS_MAP.get(kw, []))
    return list(dict.fromkeys(expanded))  # 保序去重


def locate_for_insurance(doc_index, doc_ids: list, opt_key: str,
                         opt_val: str, q_kws: list) -> tuple:
    """
    保险 multi 题：从选项文本中识别产品名 → 匹配文档 → 搜问题关键词
    返回 (label, context)
    """
    # 1. 从选项文本中找到对应文档
    target_did = None
    target_label = ""
    for did in doc_ids:
        products = INS_DOC_PRODUCTS.get(did, [])
        if any(p in opt_val for p in products):
            target_did = did
            target_label = products[0] if products else did
            break
    # 如果没匹配到，用第一个文档
    if not target_did:
        target_did = doc_ids[0] if doc_ids else None
        target_label = target_did or ""

    if not target_did:
        return ("", "")

    text = doc_index.get_doc_full_text(target_did) or ""
    if not text:
        return (target_label, "")

    # 2. 用问题关键词 + 选项特征词搜索
    # 选项中冒号后的条款特征（排除产品名）
    opt_after_colon = re.sub(r'^[^：:]{2,15}[：:]', '', opt_val).strip()
    opt_numbers = _extract_numbers(opt_after_colon)
    opt_nouns = [w for w in re.findall(r'[一-鿿]{3,8}', opt_after_colon)
                 if w not in ('允许', '不允许', '规定', '适用', '要求', '说法')]

    # 搜索优先级：问题关键词(含同义词) > 选项数字 > 选项名词
    q_kws_expanded = _expand_kws(q_kws)
    search_kws = q_kws_expanded + opt_numbers + opt_nouns[:2]
    ctx = _find_context(text, search_kws, ctx_chars=800, skip=SKIP_INS, before=200)
    if not ctx:
        # fallback: 不 skip
        ctx = _find_context(text, search_kws, ctx_chars=800, skip=0, before=200)

    return (target_label, ctx)


def locate_for_financial_contracts(doc_index, doc_ids: list,
                                   opt_key: str, opt_val: str) -> tuple:
    """
    金融合同（债券说明书）multi 题：
    - 封面（前 1200 字）包含发行人、金额、评级、承销商等所有关键字段
    - 直接返回每份文档封面，所有选项共用
    """
    parts = []
    for i, did in enumerate(doc_ids[:2]):
        text = doc_index.get_doc_full_text(did) or ""
        cover = text[:1200].strip()
        if cover:
            label = re.findall(r'[一-鿿]{4,15}', text[:200])
            label_str = label[0] if label else f"文档{i+1}"
            parts.append(f"[第{i+1}份文档: {label_str}]\n{cover}")

    return ("", "\n\n".join(parts))


def locate_for_financial_reports(doc_index, doc_ids: list,
                                  q_kws: list, opt_val: str) -> tuple:
    """
    财务报告 multi 题：
    - 财务摘要表在文档前 15K（「主要会计数据和财务指标」章节），一次性取出全表
    - 每个选项用**选项自身数字/指标名**在摘要表里对应（不重复搜文档）
    """
    opt_numbers = _extract_numbers(opt_val)
    opt_indicators = [w for w in re.findall(r'[一-鿿]{3,10}', opt_val)
                      if w not in ('同比', '增长', '下降', '减少', '增加',
                                   '较上年', '比上年', '年度', '数据')][:3]
    # 用选项数字/指标 + 问题关键词
    search_kws = opt_numbers + opt_indicators + q_kws

    parts = []
    for did in doc_ids[:2]:
        text = doc_index.get_doc_full_text(did) or ""
        if not text:
            continue
        # 找「主要会计数据和财务指标」章节（固定位置，含整张摘要表）
        anchor_kws = ['主要会计数据和财务指标', '主要财务指标', '财务数据摘要',
                      '主要会计数据', '财务摘要']
        ctx = _find_context(text, anchor_kws, ctx_chars=1400, skip=0, before=0)
        if not ctx:
            # fallback: 用选项数字在前 20K 里精确定位
            ctx = _find_context(text[:20000], search_kws, ctx_chars=1000,
                                skip=0, before=300)
        if ctx:
            label = re.findall(r'[一-鿿]{4,15}(?:报告|年报|公司)', text[:300])
            label_str = label[0] if label else did[:20]
            parts.append(f"[{label_str}]\n{ctx}")

    return ("", "\n\n".join(parts))


def locate_for_regulatory(doc_index, doc_ids: list,
                           q_kws: list, opt_val: str) -> tuple:
    """
    监管法规 multi 题：按数字+单位或条款号定位
    """
    opt_numbers = _extract_numbers(opt_val)
    opt_clauses = re.findall(r'第[一二三四五六七八九十百\d]+条', opt_val)
    opt_nouns = [w for w in re.findall(r'[一-鿿]{3,8}', opt_val)
                 if w not in ('适用', '规定', '应当', '可以', '不得')][:2]
    search_kws = opt_numbers + opt_clauses + q_kws + opt_nouns

    parts = []
    for did in doc_ids:
        text = doc_index.get_doc_full_text(did) or ""
        ctx = _find_context(text, search_kws, ctx_chars=800, skip=0, before=200)
        if ctx:
            label = did[:30]
            parts.append(f"[{label}]\n{ctx}")

    return ("", "\n\n".join(parts))


def locate_for_research(doc_index, doc_ids: list,
                         q_kws: list, opt_val: str) -> tuple:
    """
    研究报告 multi 题：**选项数字优先**，再用问题关键词
    """
    opt_numbers = _extract_numbers(opt_val)
    # 选项中的核心短语（数字+名词）
    opt_key_phrases = [w for w in re.findall(r'[一-鿿]{2,8}', opt_val)
                       if len(w) >= 3][:3]
    # 数字优先（最具区分性）
    search_kws = opt_numbers + opt_key_phrases + q_kws

    parts = []
    for did in doc_ids:
        text = doc_index.get_doc_full_text(did) or ""
        ctx = _find_context(text, search_kws, ctx_chars=800, skip=0, before=200)
        if ctx:
            label = re.findall(r'[一-鿿]{4,15}', text[:200])
            label_str = label[0] if label else did
            parts.append(f"[{label_str}]\n{ctx}")

    return ("", "\n\n".join(parts))


# ============ Prompt ============

ANSWER_PROMPT = """根据以下原文片段，严格回答多选题。

问题: {question}
选项:
{options}

原文证据（按选项整理的精准片段）:
{evidence}

判断规则:
- 原文明确支持 → 选
- 原文明确否定/与选项矛盾 → 不选
- 原文没有相关内容 → 存疑，结合已找到证据的选项综合判断
- 多选题通常有2-3个正确答案

只输出正确选项字母（如 ABC），不要解释不要分析"""


# ============ Agent ============

class ReasoningAgentV33(ReasoningAgentV31):
    """V33: 精准单窗口检索，每选项 700 字上下文，1次LLM"""

    def answer_question(self, question: dict) -> dict:
        if question.get("answer_format") == "multi":
            return self._answer_multi_v33(question)
        else:
            from agent.reasoner_v30 import ReasoningAgentV30
            return ReasoningAgentV30.answer_question(self, question)

    def _answer_multi_v33(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])
        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        system = DOMAIN_SYSTEM.get(domain, "")

        # 问题关键词（引号内 + 核心名词，严格过滤泛化词）
        STOPWORDS = {'下列哪些', '以下哪些', '关于', '说法正确', '正确的有',
                     '符合规定', '下列', '哪些', '哪个', '以下', '正确',
                     '描述准确', '说法符合', '规定的', '下列哪', '以下哪',
                     '说明', '情形', '表述', '陈述'}
        q_kws = _extract_quoted(q_text)
        if not q_kws:
            q_kws = [w for w in re.findall(r'[一-鿿]{3,10}', q_text)
                     if w not in STOPWORDS and len(w) >= 3][:4]

        # financial_contracts 特殊处理：所有选项共用两份文档封面
        fin_contracts_ctx = ""
        if domain == "financial_contracts":
            _, fin_contracts_ctx = locate_for_financial_contracts(
                self.doc_index, doc_ids, "", "")

        # financial_reports 特殊处理：先拿摘要表，所有选项共享
        fin_report_ctx = ""
        if domain == "financial_reports":
            _, fin_report_ctx = locate_for_financial_reports(
                self.doc_index, doc_ids, q_kws, "")

        # 按选项构建证据
        all_evidence_parts = []
        seen_ctxs = set()  # 去重

        for opt_key in sorted(options.keys()):
            opt_val = options[opt_key]
            ctx = ""

            if domain == "insurance":
                _, ctx = locate_for_insurance(
                    self.doc_index, doc_ids, opt_key, opt_val, q_kws)

            elif domain == "financial_contracts":
                # 共用封面，不重复
                ctx = fin_contracts_ctx

            elif domain == "financial_reports":
                # 共用摘要表，不重复拉取
                ctx = fin_report_ctx

            elif domain == "regulatory":
                _, ctx = locate_for_regulatory(
                    self.doc_index, doc_ids, q_kws, opt_val)

            elif domain == "research":
                _, ctx = locate_for_research(
                    self.doc_index, doc_ids, q_kws, opt_val)

            if ctx and ctx not in seen_ctxs:
                seen_ctxs.add(ctx)
                all_evidence_parts.append(f"--- 选项{opt_key} ---\n{ctx}")

        total_evidence_chars = sum(len(p) for p in all_evidence_parts)
        combined_evidence = "\n\n".join(all_evidence_parts)

        # 安全网：证据太少 → 回退 V22
        if total_evidence_chars < 400:
            print(f" [FALLBACK V22: ev={total_evidence_chars}c]")
            from agent.reasoner_v22 import ReasoningAgentV22
            return ReasoningAgentV22.answer_question(self, question)

        answer_prompt = ANSWER_PROMPT.format(
            question=q_text,
            options=options_text,
            evidence=combined_evidence[:5000],
        )
        answer_raw = ""
        try:
            result = self.qwen.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": answer_prompt}],
                temperature=0.1, max_tokens=64, timeout=90,
                enable_thinking=False)
            answer_raw = result["content"]
        except Exception as e:
            print(f" [ANS ERR:{e}]")

        answer = extract_answer_from_response(answer_raw, "multi")
        answer = self._post_process(answer, "multi")

        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": answer,
            "answer_format": "multi",
            "evidence_chars": total_evidence_chars,
            "total_doc_chars": total_doc_chars,
            "is_full_doc": False,
            "raw_response": (
                f"[q_kws={q_kws}]\n\n"
                f"[evidence={total_evidence_chars}c]\n{combined_evidence[:2000]}\n\n"
                f"[answer_raw]\n{answer_raw}"
            )[:3000],
        })

        return {"qid": qid, "answer": answer,
                "evidence_chars": total_evidence_chars,
                "total_doc_chars": total_doc_chars}

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v33.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = [dict(t, raw_response=t.get("raw_response", "")[:2000]) for t in self.cot_trails]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  [OK] COT trails -> {path}")
