"""V22: V20 + fc 锚词扩展 + 财报锚词扩展.

V20 = 43.81 是基线. V22 在 V21 (仅 fc 扩展) 基础上, 加上 财报锚词扩展.

改动 1 (V21): fc 锚词从 5 个扩到 26 个
改动 2 (V22 新): 财报除 "主要会计数据/现金流" 外, 加锚词:
- 研发投入, 研发费用 (V20 fin_a_004 漏了 midea 2025 研发投入 @ 27.8K)
- 股份回购, 回购 (V20 fin_a_004 漏了 midea 2025 回购 @ 59.2K)
- 派发现金 (分红方案)

不动: 其他域 (research/insurance/regulatory), prompt 模板, 后处理.
"""
from agent.reasoner_v21 import ReasoningAgentV21, build_evidence_v21, V21_DOMAIN_EXTRACTORS
from agent.reasoner_v20 import _take_head, _locate_section, DOMAIN_SYSTEM, PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI


def extract_evidence_financial_report_v22(text: str, max_chars: int = 50000) -> str:
    """V22 财报: 前 25K + 主要会计数据 + 现金流/分红 + 研发/回购"""
    head = _take_head(text, 25000)
    seen = set()

    # 1. 主要会计数据章节
    key_section = _locate_section(text, ["主要会计数据", "主要财务指标"], ctx_chars=10000)
    if key_section:
        k = key_section[:50]
        if k not in seen and k not in head:
            seen.add(k)
            head += "\n[主要会计数据章节]\n" + key_section
            if len(head) >= max_chars:
                return head[:max_chars]

    # 2. 现金流/分红
    cash_section = _locate_section(text, ["现金流量净额", "现金分红", "利润分配", "派发现金"], ctx_chars=6000)
    if cash_section:
        k = cash_section[:50]
        if k not in seen and k not in head:
            seen.add(k)
            head += "\n[现金流/分红章节]\n" + cash_section
            if len(head) >= max_chars:
                return head[:max_chars]

    # 3. V22 新增: 研发投入 / 研发费用
    for anchor in ["研发投入", "研发费用"]:
        sec = _locate_section(text, [anchor], ctx_chars=4000)
        if sec:
            k = sec[:50]
            if k in seen or k in head:
                continue
            seen.add(k)
            head += f"\n[{anchor}]\n" + sec
            if len(head) >= max_chars:
                return head[:max_chars]

    # 4. V22 新增: 股份回购 / 回购
    for anchor in ["股份回购", "回购"]:
        sec = _locate_section(text, [anchor], ctx_chars=4000)
        if sec:
            k = sec[:50]
            if k in seen or k in head:
                continue
            seen.add(k)
            head += f"\n[{anchor}]\n" + sec
            if len(head) >= max_chars:
                return head[:max_chars]

    return head[:max_chars]


# 复用 V21 的 fc 扩展, V22 加财报扩展
V22_DOMAIN_EXTRACTORS = dict(V21_DOMAIN_EXTRACTORS)
V22_DOMAIN_EXTRACTORS["financial_reports"] = extract_evidence_financial_report_v22


def build_evidence_v22(doc_index, domain, doc_ids, n_docs, max_evidence):
    """证据组装"""
    from agent.reasoner_v20 import extract_evidence_regulatory
    extractor = V22_DOMAIN_EXTRACTORS.get(domain, extract_evidence_regulatory)
    per_doc = max_evidence // max(1, n_docs)
    evidence = ""
    for did in doc_ids:
        t = doc_index.get_doc_full_text(did) or ""
        if not t:
            continue
        seg = extractor(t, max_chars=per_doc)
        evidence += f"\n=== 文档 {did} ===\n{seg}\n"
    return evidence


class ReasoningAgentV22(ReasoningAgentV21):
    """V22: V21 + 财报锚词扩展"""

    def answer_question(self, question: dict) -> dict:
        from agent.postprocessor import extract_answer_from_response
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        max_ev = self.MAX_EVIDENCE.get(answer_format, 80000)

        evidence = build_evidence_v22(self.doc_index, domain, doc_ids, len(doc_ids), max_ev)

        if answer_format == "tf":
            prompt_tpl = PROMPT_TF
        elif answer_format == "mcq":
            prompt_tpl = PROMPT_MCQ
        else:
            prompt_tpl = PROMPT_MULTI

        prompt = prompt_tpl.format(
            evidence=evidence, question=q_text,
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
        )

        system = DOMAIN_SYSTEM.get(domain, "")

        try:
            result = self.qwen.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""

        answer = extract_answer_from_response(raw_response, answer_format)
        answer = self._post_process(answer, answer_format)

        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": answer,
            "answer_format": answer_format,
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": total_doc_chars <= max_ev,
            "raw_response": raw_response,
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
        }

    def save_cot_trails(self, path=None):
        import os, json
        from agent.config import RESULTS_DIR
        path = path or os.path.join(RESULTS_DIR, "eval_results_v22.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = []
        for t in self.cot_trails:
            t2 = dict(t)
            if "raw_response" in t2:
                t2["raw_response"] = t2["raw_response"][:1500]
            out.append(t2)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  ✅ COT trails -> {path}")
