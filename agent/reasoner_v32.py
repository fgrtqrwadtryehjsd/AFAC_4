"""V32: 纯正则关键词提取 + 选项导向定位 + 直接回答

v1-v8 失败教训:
- LLM 关键词提取不稳定: 提取产品名、泛化词、超时
- BM25 对中文 query 无效
- compress 步骤丢失否定信息

v9 新架构:
  Step 0 (零LLM): 纯正则从问题和选项提取关键词
    - 问题关键词: 引号内词 + 核心名词短语 (无LLM)
    - 选项关键词: 每选项的数字/短条款名 (无LLM)
    - 保险domain: 额外按选项中产品名匹配对应文档
  Step 1: 批量全文定位 (零LLM)
  Step 2: 一次LLM回答

tf/mcq: 保留 V31 路径 (V30精炼)
"""
import re
import json
import os
from agent.indexer import DocumentIndex
from agent.reasoner_v31 import ReasoningAgentV31
from agent.reasoner_v20 import DOMAIN_SYSTEM
from agent.postprocessor import extract_answer_from_response
from agent.config import RESULTS_DIR


# ============ 检索工具 ============

SKIP_PREFIX = {
    "insurance": 8000,
    "financial_contracts": 0,
    "financial_reports": 0,
    "regulatory": 0,
    "research": 0,
}

# 泛化词黑名单（不提取为关键词）
KW_STOPWORDS = {
    '下列哪些', '以下哪些', '下列哪个', '以下哪个', '说法正确', '正确的有',
    '符合规定', '不正确', '不符合', '相关规定', '关于', '哪些', '哪个',
    '规定', '允许', '要求', '适用', '不得', '应当', '可以',
}


def extract_question_keywords(q_text: str) -> list:
    """从问题文本纯正则提取核心关键词（无LLM）"""
    # 1. 引号内的词（最高优先级）
    quoted = re.findall(r'["""«»「」『』【】《》](.*?)["""»「」』】》]', q_text)
    if quoted:
        return [w for w in quoted[:3] if len(w) >= 2]
    # 2. 关键名词短语（3-8字中文）
    words = re.findall(r'[一-鿿]{3,8}', q_text)
    filtered = [w for w in words if w not in KW_STOPWORDS]
    return filtered[:3]


def extract_option_keywords(opt_val: str) -> list:
    """从选项文本纯正则提取区分性关键词（无LLM）"""
    kws = []
    # 1. 数字+单位（优先，最具区分性）
    nums = re.findall(
        r'\d+(?:\.\d+)?(?:\s*(?:%|亿|万|元|年|月|日|天|个?月|个?工作日|条|款))',
        opt_val)
    kws.extend([n.replace(' ', '') for n in nums[:2]])
    # 2. 条文号（第X条）
    clauses = re.findall(r'第[一二三四五六七八九十百]+条|第\d+条', opt_val)
    kws.extend(clauses[:1])
    # 3. 冒号后的首个名词短语（保险选项如 "平安智盈金生：..."）
    m = re.match(r'[^：:，,。！]{2,10}[：:]\s*(.{3,12})', opt_val)
    if m:
        w = re.findall(r'[一-鿿]{3,8}', m.group(1))
        kws.extend([x for x in w if x not in KW_STOPWORDS][:1])
    # 4. 纯中文短语 fallback
    if not kws:
        words = re.findall(r'[一-鿿]{3,8}', opt_val)
        kws.extend([w for w in words if w not in KW_STOPWORDS][:2])
    return kws[:3]


def locate_keyword_in_doc(text: str, keyword: str, ctx_chars: int = 1200,
                           skip: int = 0) -> str:
    """在文档全文中定位关键词，返回上下文"""
    search_text = text[skip:]
    pos = search_text.find(keyword)
    if pos < 0:
        return ""
    abs_pos = skip + pos
    start = max(0, abs_pos - 200)
    return text[start: start + ctx_chars]


def locate_in_doc(text: str, keywords: list, domain: str, ctx_chars: int = 1200) -> str:
    """在单个文档中搜索关键词列表，返回第一个找到的上下文"""
    skip = SKIP_PREFIX.get(domain, 0)
    for kw in keywords:
        seg = locate_keyword_in_doc(text, kw, ctx_chars, skip)
        if seg:
            return seg
        # 同义词
        for syn in _get_synonyms(kw):
            seg = locate_keyword_in_doc(text, syn, ctx_chars, skip)
            if seg:
                return f"[同义词'{syn}']\n{seg}"
    return ""


def _get_synonyms(keyword: str) -> list:
    SYNONYMS = {
        "保单贷款": ["借款", "保单借款"],
        "借款": ["保单贷款", "保单借款"],
        "营业收入": ["营收", "营业总收入"],
        "归母净利润": ["归属于母公司净利润", "归属母公司净利润"],
        "研发投入": ["研发费用", "研究开发费用"],
        "股份回购": ["回购", "回购股份"],
        "现金分红": ["利润分配", "派发现金股利"],
        "身故保险金": ["死亡保险金", "身故给付"],
        "现金价值": ["保单现金价值", "退保现金价值"],
        "犹豫期": ["犹豫撤单", "15天", "10天"],
        "施行日期": ["施行", "生效日期", "生效时间"],
    }
    return SYNONYMS.get(keyword, [])


def get_doc_label(text: str, did: str) -> str:
    """从文档前200字提取产品/文档标签"""
    head = text[:300]
    # 跳过页码行，找第一个 ≥4字的中文短语
    m = re.search(r'[一-鿿]{4,25}', head[20:])
    return m.group() if m else did


# ============ Prompt ============

ANSWER_PROMPT = """根据检索到的原文片段回答多选题。

问题: {question}
选项:
{options}

原文证据（按选项分组）:
{evidence}

规则:
- 原文明确支持选项内容 → 选
- 原文明确否定选项内容 → 不选
- 未检索到相关内容 → 结合其他选项综合判断（多选题通常有2-3个正确答案）
- 不能输出空，至少选1个

直接输出正确选项字母（如 ABC），不要解释"""


# ============ Agent ============

class ReasoningAgentV32(ReasoningAgentV31):
    """V32: 纯正则关键词提取 + 一次LLM回答 (enable_thinking=False)"""

    def answer_question(self, question: dict) -> dict:
        answer_format = question.get("answer_format", "mcq")
        domain = question.get("domain", "")
        if answer_format == "multi":
            # 只对法规类 multi 使用关键词定位（条文明确、关键词精准）
            # 其他 domain 走 V22 全文路径（保险细粒度判断、财报数字精确）
            if domain == "regulatory":
                return self._answer_multi_v32(question)
            from agent.reasoner_v22 import ReasoningAgentV22
            return ReasoningAgentV22.answer_question(self, question)
        else:
            from agent.reasoner_v30 import ReasoningAgentV30
            return ReasoningAgentV30.answer_question(self, question)

    def _answer_multi_v32(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])
        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        system = DOMAIN_SYSTEM.get(domain, "")

        # ---- Step 0: 零LLM关键词提取 ----
        q_kws = extract_question_keywords(q_text)  # 问题核心词
        opt_kws = {}
        for opt_key, opt_val in options.items():
            per_opt = extract_option_keywords(opt_val)
            opt_kws[opt_key] = per_opt

        # ---- Step 1: 批量定位 ----
        # 预加载所有文档文本
        docs = {}
        doc_labels = {}
        for did in doc_ids:
            text = self.doc_index.get_doc_full_text(did) or ""
            docs[did] = text
            doc_labels[did] = get_doc_label(text, did)

        total_evidence_chars = 0
        all_evidence_parts = []

        for opt_key in sorted(options.keys()):
            opt_val = options[opt_key]
            opt_parts = []

            # 保险 domain 特殊逻辑: 识别选项中的产品名 → 只搜对应文档
            if domain == "insurance":
                # 找选项提到的产品名对应的文档
                target_did = None
                for did in doc_ids:
                    label = doc_labels[did]
                    label_words = re.findall(r'[一-鿿]{3,}', label)
                    if any(w in opt_val for w in label_words if len(w) >= 3):
                        target_did = did
                        break
                search_doc_ids = [target_did] if target_did else doc_ids
                # 在目标文档中搜问题关键词和选项专属关键词
                kws_to_try = opt_kws.get(opt_key, []) + q_kws
                for did in search_doc_ids:
                    seg = locate_in_doc(docs[did], kws_to_try, domain, ctx_chars=1000)
                    if seg:
                        opt_parts.append(f"[{doc_labels[did]}]\n{seg}")
            else:
                # 策略A: 用选项专属关键词搜所有文档
                for kw in opt_kws.get(opt_key, [])[:2]:
                    for did in doc_ids:
                        seg = locate_in_doc(docs[did], [kw], domain, ctx_chars=900)
                        if seg:
                            opt_parts.append(f"[{doc_labels[did]}|kw:{kw}]\n{seg}")
                            break  # 每个关键词只取第一个文档

                # 策略B: 用问题关键词搜每个文档（每文档一条）
                if not opt_parts:
                    for did in doc_ids:
                        seg = locate_in_doc(docs[did], q_kws, domain, ctx_chars=800)
                        if seg:
                            opt_parts.append(f"[{doc_labels[did]}]\n{seg}")

            if opt_parts:
                combined_opt = "\n\n".join(opt_parts)
                total_evidence_chars += len(combined_opt)
                all_evidence_parts.append(f"--- 选项{opt_key} ---\n{combined_opt}")

        combined_evidence = "\n\n".join(all_evidence_parts)

        # 安全网: 证据过少 → 用问题关键词搜全部文档作为 fallback evidence
        if total_evidence_chars < 500:
            fallback_parts = []
            for did in doc_ids:
                seg = locate_in_doc(docs[did], q_kws, domain, ctx_chars=1200)
                if seg:
                    fallback_parts.append(f"[{doc_labels[did]}]\n{seg}")
            combined_evidence = "\n\n".join(fallback_parts)
            total_evidence_chars = len(combined_evidence)

        # 如果还是没有证据 → 回退 V22
        if total_evidence_chars < 200:
            print(f" [FALLBACK V22: ev={total_evidence_chars}]")
            from agent.reasoner_v22 import ReasoningAgentV22
            return ReasoningAgentV22.answer_question(self, question)

        # ---- Step 2: 一次LLM回答 ----
        answer_prompt = ANSWER_PROMPT.format(
            question=q_text,
            options=options_text,
            evidence=combined_evidence[:6000],
        )
        answer_raw = ""
        try:
            result = self.qwen.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": answer_prompt}],
                temperature=0.1, max_tokens=256, timeout=90,
                enable_thinking=False)
            answer_raw = result["content"]
        except Exception as e:
            print(f" [ANS ERR:{e}]")

        answer = extract_answer_from_response(answer_raw, "multi")
        answer = self._post_process(answer, "multi")

        kw_summary = f"q_kws={q_kws} | opt_kws={opt_kws}"
        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": answer,
            "answer_format": "multi",
            "evidence_chars": total_evidence_chars,
            "total_doc_chars": total_doc_chars,
            "is_full_doc": False,
            "raw_response": (
                f"[kws] {kw_summary}\n\n"
                f"[evidence={total_evidence_chars}c]\n{combined_evidence[:1500]}\n\n"
                f"[answer_raw]\n{answer_raw}"
            )[:3000],
        })

        return {"qid": qid, "answer": answer,
                "evidence_chars": total_evidence_chars,
                "total_doc_chars": total_doc_chars}

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v32.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = [dict(t, raw_response=t.get("raw_response", "")[:2000]) for t in self.cot_trails]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  [OK] COT trails -> {path}")
