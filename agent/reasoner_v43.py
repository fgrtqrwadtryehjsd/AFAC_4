"""V43: V42 + tf/mcq 靶向证据覆盖修复 + 宽容prompt

## 根因 (2026-07-10 小样本验证)
V42 tf/mcq 走 V30 窄结构化证据(8-14K), 截掉深位事实(43.24%@text10深位51168, 2019回购@midea深位54159,
宁德时代市占率等) → 模型看不到证据 → 判错. tf/mcq 21/35 一致 (限速器), multi 54/65 (V35广域60K).
真瓶颈 = tf/mcq 证据覆盖, 非 prompt 宽容度 (宽容alone仅+3/10).

## 修法 (验证 +9 tf, 0回归)
tf/mcq: V30基础 + 靶向grep陈述/选项事实(补深位) + 宽容PROMPT_TF/MCQ. multi不变(V35+审计).
- 关键词提取无分词器: 文档锚定最长CJK匹配(每位置取原文真实存在的最长子串, 自动发现实体, 不产碎片)
- 数字须带单位/小数/% (滤doc-id 001/004)
- grep频率感知: 稀有词(低频)先取, 泛词(高频)后取, 轮询每词≥1条, 防洪泛
- 子串去重: A是B子串则去A, 消除冗余碎片(期债券发行/债券发行/券发行)腾slot给稀有数(150%)

## 小样本验证
- tf: 10翻转+5ctrl → 9/10修对(res_a_006实为V43更对, test_A 8错之一), 0回归
- mcq: 4diff+5ctrl → 1/4修对(res_a_008), 0回归; 余3为推理/陷阱/计算非覆盖
- 全量预估: V42 73对 → V43 +9~10 → 82~83对, +~175K token → 3.4M, factor0.796 → ~66
"""
import re
from agent.reasoner_v30 import build_evidence_v30
from agent.reasoner_v20 import DOMAIN_SYSTEM
from agent.postprocessor import extract_answer_from_response
from agent.kg_reasoner import ReasoningAgentV42


_STOP2 = {"文档", "公司", "报告", "条款", "文件", "数据", "信息", "以下", "是否",
          "正确", "判断", "陈述", "上述", "根据", "提供", "一份", "两份", "第二",
          "第一", "本文", "本报告", "以上", "如下", "其中", "以及", "并且", "通过",
          "进行", "可以", "应当", "不得", "已经", "下列", "描述", "哪一", "哪项"}


def text_terms(text, doc_texts):
    """数字(须带单位/小数/%) + 文档锚定最长CJK匹配 + 子串去重."""
    raw_nums = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:%|亿|万|元|倍|股|年|个月|个工作日|日|月)?", text)
    nums = []
    for n in raw_nums:
        n = n.strip()
        if len(re.sub(r"\D", "", n)) < 2:
            continue
        has_unit = bool(re.search(r"%|亿|万|元|倍|股|年|个月|日|月", n))
        has_dec = ("." in n) or ("%" in n)
        if has_unit or has_dec:
            nums.append(n)
    nums = list(dict.fromkeys(nums))
    terms = set()
    for i in range(len(text)):
        for L in range(min(8, len(text) - i), 1, -1):
            sub = text[i:i + L]
            if not re.fullmatch(r"[一-鿿]+", sub):
                continue
            if any(sub in t for t in doc_texts):
                terms.add(sub)
                break
    terms = {t for t in terms if not (len(t) == 2 and t in _STOP2)}
    # 子串去重: A是B子串则去A(保长), 消除冗余碎片
    terms_list = sorted(terms, key=lambda x: (-len(x), x))
    dedup = []
    for t in terms_list:
        if not any(t != o and t in o for o in terms):
            dedup.append(t)
    return nums, dedup


def targeted_grep(text, search_kws, max_chars=6000, per_kw_cap=1):
    """频率感知grep: 稀有词先取, 轮询每词≥1条(per_kw_cap=1最大化多样性), 防洪泛."""
    if not search_kws: return ""
    kw_pos, kw_freq = {}, {}
    for kw in search_kws:
        if not kw: continue
        ps = [m.start() for m in re.finditer(re.escape(kw), text)]
        if ps:
            kw_pos[kw] = ps[:per_kw_cap]
            kw_freq[kw] = len(ps)
    if not kw_pos: return ""
    kws_sorted = sorted(kw_pos.keys(), key=lambda k: (kw_freq[k], -len(k), k))
    chosen, used = [], []
    for rnd in range(per_kw_cap):
        for kw in kws_sorted:
            if len(kw_pos[kw]) <= rnd: continue
            pos = kw_pos[kw][rnd]
            if any(abs(pos - p) < 350 for p in used): continue
            chosen.append((pos, kw)); used.append(pos)
    chosen.sort(key=lambda x: (kw_freq[x[1]], x[0]))
    snippets = []
    for pos, kw in chosen:
        if sum(len(s) for s in snippets) >= max_chars: break
        s, e = max(0, pos - 300), min(len(text), pos + 550)
        snippets.append(f'...{text[s:e].strip()}...')
    return "\n---\n".join(snippets)[:max_chars]


def build_evidence_v43(di, q, answer_format, max_base=50000, max_target_per_doc=6000):
    """V30基础 + 靶向grep(tf=陈述事实; mcq=所有选项事实并集)."""
    domain = q.get("domain", "")
    doc_ids = q.get("doc_ids", [])
    options = q.get("options", {})
    base = build_evidence_v30(di, domain, doc_ids, max_base)
    doc_texts = [di.get_doc_full_text(str(d)) or "" for d in doc_ids]
    if answer_format == "tf":
        nums, terms = text_terms(q.get("question", ""), doc_texts)
        search_kws = nums[:8] + terms[:18]
    else:  # mcq: 并集所有选项
        all_nums, all_terms = [], []
        for ok in sorted(options):
            n, t = text_terms(options[ok], doc_texts)
            all_nums += n
            all_terms += t
        search_kws = list(dict.fromkeys(all_nums))[:10] + list(dict.fromkeys(all_terms))[:20]
    targeted = ""
    for did, t in zip(doc_ids, doc_texts):
        if not t: continue
        seg = targeted_grep(t, search_kws, max_target_per_doc)
        if seg:
            targeted += f"\n=== 文档 {did} 靶向补充 ===\n{seg}\n"
    return base + targeted, len(base), len(targeted)


LENIENT_PROMPT_TF = """你的任务: 依据下列文档证据判断陈述是否正确. 命题人按语义判定: 核心事实/数值/主体/方向与原文一致即正确, 不要求用词逐字一致.

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 判断流程
1. 把陈述拆成可独立核验的子陈述 (按"且"/"同时"/";"/","拆分).
2. 对每个子陈述, 在原文找语义对应语句 (允许同义改述/等价表述/概括, 不要求逐字).
3. 区分两类差异:
   硬矛盾(否决, 判B): 数值/百分比/倍数不符; 年份张冠李戴; 主体错位(A公司的事说成B); 方向相反(上升说成下降); 时限不一致(6个月说成3个月); 单位错位.
   软差异(容忍, 不否决): 同义用词(如"可疑"="大额"当语境等价); 等价表述(如"及时、公平地履行"="及时履行"); 概括("保护性条款"涵盖"回售选择权""赎回安排"); 默认指代.
4. 判定: 所有子陈述均无硬矛盾(可有软差异) -> A 正确; 任一子陈述有硬矛盾 -> B 错误.
5. 仅当存在实质性事实冲突(数值/主体/方向/年份/时限/单位)才判错; 用词不完全一致但语义一致 -> 正确.

## 输出格式 (必须遵守)
子陈述1: <拆分> | 原文核验: <引用> | 硬矛盾(无/有:类型) | 判定(✓/✗)
子陈述2: ...
最终答案: A 或 B
"""

LENIENT_PROMPT_MCQ = """你的任务: 依据下列文档证据选出最直接被原文支持的选项. 命题人按语义判定: 核心事实/数值/主体与原文一致即支持, 不要求用词逐字一致.

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 判断流程
1. 对 A B C D 4 个选项全部分析, 不能只看 A 就停.
2. 每个选项在原文找语义对应语句 (允许同义改述/等价表述/概括, 不要求逐字), 比较核心数值/主体/方向.
3. 硬矛盾(否决该选项): 数值/百分比/倍数不符; 主体错位(A公司说成B); 方向相反; 年份张冠李戴; 单位错位(每股vs每10股); 时限不一致.
4. 软差异(不否决): 同义用词/等价表述/概括/默认指代.
5. 选: 无硬矛盾且最直接被原文支持的选项. 警惕数值/主体/方向/单位陷阱, 但不苛求用词逐字一致.

## 输出格式 (必须遵守)
选项 A: <原文引用> | 硬矛盾(无/有:类型) | 判定(✓/✗)
选项 B: ...
选项 C: ...
选项 D: ...
最终答案: A 或 B 或 C 或 D
"""

LENIENT_DOMAIN_SYSTEM = {
    "insurance": DOMAIN_SYSTEM["insurance"],
    "regulatory": "你是金融监管合规专家. 关键: \"应当\"\"必须\"\"不得\"=强制; \"可以\"=授权; \"大额交易报告\"≠\"可疑交易报告\"; 时限 (30 个工作日/6 个月/10 年) 须精确匹配原文. 实质性事实冲突(数值/主体/方向/时限)才判错, 同义改述不判错.",
    "financial_contracts": "你是金融合同分析师. 关键: 主体信用评级 ≠ 债项信用评级; 第一份文档与第二份文档必须分别核验, 不能张冠李戴. 数字/评级/期限与原文语义一致即可, 不要求逐字.",
    "financial_reports": DOMAIN_SYSTEM["financial_reports"],
    "research": DOMAIN_SYSTEM["research"],
}


class ReasoningAgentV43(ReasoningAgentV42):
    """V43: multi 走 V42(V35+审计); tf/mcq 走 V30基础+靶向grep+宽容prompt."""

    def answer_question(self, question):
        af = question.get("answer_format", "mcq")
        if af == "multi":
            return self._answer_multi_v42(question)
        return self._answer_tfmq_v43(question)

    def _answer_tfmq_v43(self, question):
        qid = question["qid"]
        domain = question.get("domain", "")
        af = question.get("answer_format", "mcq")
        options = question.get("options", {})
        evidence, nbase, ntgt = build_evidence_v43(self.di, question, af)
        prompt_tpl = LENIENT_PROMPT_TF if af == "tf" else LENIENT_PROMPT_MCQ
        prompt = prompt_tpl.format(
            evidence=evidence, question=question["question"],
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())))
        system = LENIENT_DOMAIN_SYSTEM.get(domain, "")
        try:
            r = self.qwen.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, timeout=180)
            raw = r["content"]
        except Exception as e:
            raw = f"ERR:{e}"
        ans = extract_answer_from_response(raw, af)
        # 后处理: tf→AB, mcq→ABCD单
        chars = sorted(set(c for c in (ans or "").upper() if c in "ABCD"))
        if af == "tf":
            ans = "".join(chars) if chars else ""
            if not ans:
                ans = "A"
        else:
            ans = chars[0] if chars else "A"
        total_doc_chars = sum(self.di.doc_lengths.get(str(d), 0) for d in question.get("doc_ids", []))
        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": ans, "answer_format": af,
            "evidence_chars": len(evidence), "base_chars": nbase, "target_chars": ntgt,
            "total_doc_chars": total_doc_chars, "raw_response": (raw or "")[:1500],
            "strategy": "v43_tfmq_targeted_lenient",
        })
        return {"qid": qid, "answer": ans,
                "evidence_chars": len(evidence), "total_doc_chars": total_doc_chars}
