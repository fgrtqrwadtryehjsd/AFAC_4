"""多轮验证Agent: 定位→推理→验证(漏选检查)
模拟Claude做ground truth的方法: grep定位+逐选项推理+反向查漏选

Stage1+2 混合证据: 精准grep(60%) + head保底(40%), 不漏信息
Stage3 推理: 复用V20 prompt, 逐选项判断
Stage4 验证(仅multi): 对draft未选选项, 专门grep定位+Qwen判断是否有原文支持, 修漏选
"""
import re, os, json
from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI, _take_head
from agent.postprocessor import extract_answer_from_response
from agent.config import RESULTS_DIR


def extract_keywords(text, max_kw=40):
    kws = set()
    for m in re.finditer(r'\d[\d,\.]*\s*(?:亿|万|元|%|股|倍|年|个工作日|个月|日|期|元/股)', text):
        kws.add(m.group().strip())
    for m in re.finditer(r'\d{4}年', text):
        kws.add(m.group())
    for m in re.finditer(r'["“]([^"”]{2,20})["”]', text):
        kws.add(m.group(1))
    for m in re.finditer(r'[一-龥]{3,8}', text):
        kws.add(m.group())
    stop = {'不超过','下列','以下','关于','根据','结合','正确的','描述','选项','文档','内容','情况',
            '哪些','说法','本期','公司','下列说法','正确的有','以下说法','属于','相关','具体','明确'}
    kws = {k for k in kws if len(k) >= 2 and k not in stop}
    return list(kws)[:max_kw]


def locate_sections(text, keywords, max_chars):
    positions = []
    for kw in keywords:
        for m in re.finditer(re.escape(kw), text):
            positions.append(m.start())
    positions.sort()
    sections = []
    i = 0
    while i < len(positions) and sum(len(s) for s in sections) < max_chars:
        start = max(0, positions[i] - 600)
        end = min(len(text), positions[i] + 1400)
        j = i + 1
        while j < len(positions) and positions[j] - positions[j-1] < 1500:
            end = min(len(text), positions[j] + 1400)
            j += 1
        sec = text[start:end]
        if not any(_sim(sec, s) for s in sections):
            sections.append(sec)
        i = j
    return "\n---\n".join(sections)[:max_chars]


def _sim(a, b, thr=0.4):
    la, lb = len(a), len(b)
    if la == 0 or lb == 0: return False
    overlap = len(set(a[:300]) & set(b[:300]))
    return overlap / min(30, min(la, lb)//10) > thr


class MultiRoundAgent:
    def __init__(self, qwen, doc_index, token_budget=5000000):
        self.qwen = qwen
        self.doc_index = doc_index
        self.token_budget = token_budget
        self.cot_trails = []

    def _build_evidence(self, question, max_chars=20000):
        q_text = question["question"]
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])
        all_kw = extract_keywords(q_text + " " + " ".join(str(v) for v in options.values()))
        parts = []
        for did in doc_ids:
            text = self.doc_index.get_doc_full_text(did) or ""
            if not text: continue
            precise_budget = int(max_chars * 0.6)
            precise = locate_sections(text, all_kw, precise_budget)
            head_budget = max_chars - len(precise)
            head = _take_head(text, head_budget) if head_budget > 2000 else ""
            seg = (precise + "\n---HEAD---\n" + head) if head else precise
            parts.append(f"=== 文档 {did} ===\n{seg}")
        return "\n\n".join(parts)

    def _reason(self, question, evidence):
        af = question.get("answer_format", "mcq")
        q_text = question["question"]
        options = question.get("options", {})
        domain = question.get("domain", "")
        tpl = {"tf": PROMPT_TF, "mcq": PROMPT_MCQ}.get(af, PROMPT_MULTI)
        prompt = tpl.format(
            evidence=evidence, question=q_text,
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
        )
        system = DOMAIN_SYSTEM.get(domain, "")
        try:
            r = self.qwen.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, timeout=180)
            raw = r["content"]
        except Exception as e:
            raw = ""
        return extract_answer_from_response(raw, af), raw

    def _verify_multi(self, question, evidence, draft):
        options = question.get("options", {})
        draft_opts = set(draft)
        unselected = set(options.keys()) - draft_opts
        if not unselected: return draft
        # 对未选选项专门grep定位
        extra = ""
        for o in sorted(unselected):
            opt_kws = extract_keywords(options.get(o, ""))
            for did in question.get("doc_ids", []):
                text = self.doc_index.get_doc_full_text(did) or ""
                sec = locate_sections(text, opt_kws, 3000)
                if sec:
                    extra += f"\n[选项{o}定位证据]\n{sec}"
        if not extra: return draft
        verify_prompt = f"""初步答案: {draft}
已选选项: {sorted(draft_opts)}
未选选项: {sorted(unselected)}

## 补充证据(针对未选选项定向定位)
{extra}

## 原始证据(节选)
{evidence[:6000]}

## 任务
对每个未选选项, 严格判断: 原文中是否存在与该选项关键数字/用词/事实完全一致的语句?
- 有明确原文支持(数字/用词逐字匹配) → 应补充
- 无明确支持(数字微差/用词替换/主体错位/绝对化) → 不补充
- 宁可漏选, 不可过选

## 选项原文
""" + "\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())) + """

## 输出格式
未选选项X: <原文引用或"无">, 判定: 补充/不补充
...
最终答案: <按字母序>
"""
        system = DOMAIN_SYSTEM.get(question.get("domain", ""), "")
        try:
            r = self.qwen.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": verify_prompt}],
                temperature=0.1, max_tokens=2048, timeout=120)
            raw = r["content"]
        except:
            raw = ""
        ans = extract_answer_from_response(raw, "multi")
        return ans if ans else draft

    def _post(self, answer, af):
        if not answer: return "A"
        answer = answer.upper().strip()
        chars = sorted(set(c for c in answer if c in "ABCD"))
        return "".join(chars) if chars else "A"

    def answer_question(self, question):
        af = question.get("answer_format", "mcq")
        evidence = self._build_evidence(question)
        draft, raw1 = self._reason(question, evidence)
        final = draft
        if af == "multi" and draft:
            verified = self._verify_multi(question, evidence, draft)
            final = verified
        final = self._post(final, af)
        self.cot_trails.append({
            "qid": question["qid"], "draft": draft, "final": final,
            "evidence_chars": len(evidence), "raw_reason": raw1[:1200],
        })
        return {"qid": question["qid"], "answer": final, "evidence_chars": len(evidence)}

    def save_cot(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_multiround.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)
        print(f"  COT -> {path}")
