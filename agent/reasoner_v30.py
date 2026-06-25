"""V30: 结构化预提取架构 — 大幅降低 Token 消耗

V22: 4.25M token, TokenScore=0.149, Acc=60%, Score=44.69
V30: ~560K token, TokenScore=0.888, 若 Acc≥57% → Score≥55 (+10分以上)

核心改变: 每文档使用"结构化提取"代替"截取大段原文"
- 财报:主要会计数据表 + 研发 + 分红段(8K/双doc)
- 合同:发行概况 + 评级 + 关键条款(9K/双doc)
- 保险:产品名 + 关键条款(13K/4doc)
- 法规:原文(29K,本来就短)
- 研报:投资要点 + 高密度数据段(11K/双doc)
"""
from agent.reasoner_v22 import ReasoningAgentV22
from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI
from agent.structured_evidence import STRUCTURED_EXTRACTORS
from agent.postprocessor import extract_answer_from_response


def build_evidence_v30(doc_index, domain, doc_ids, max_evidence=90000):
    """结构化证据组装 — 每文档用域专属提取器"""
    extractor = STRUCTURED_EXTRACTORS.get(domain)
    if not extractor:
        # fallback
        from agent.reasoner_v20 import extract_evidence_regulatory
        extractor = extract_evidence_regulatory

    per_doc = max_evidence // max(1, len(doc_ids))
    evidence = ""
    for did in doc_ids:
        t = doc_index.get_doc_full_text(did) or ""
        if not t:
            continue
        seg = extractor(t)
        # 对每 doc 单独的提取不超过 per_doc
        seg = seg[:per_doc]
        evidence += f"\n=== 文档 {did} ===\n{seg}\n"
    return evidence


class ReasoningAgentV30(ReasoningAgentV22):
    """V30: 结构化预提取"""

    # 证据上限可以小很多 — 因为提取后内容精炼
    MAX_EVIDENCE = {
        "tf": 50000,
        "mcq": 50000,
        "multi": 50000,
    }

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        max_ev = self.MAX_EVIDENCE.get(answer_format, 50000)

        # V30 核心: 结构化证据
        evidence = build_evidence_v30(self.doc_index, domain, doc_ids, max_ev)

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
            "is_full_doc": False,
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
        path = path or os.path.join(RESULTS_DIR, "eval_results_v30.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = []
        for t in self.cot_trails:
            t2 = dict(t)
            if "raw_response" in t2:
                t2["raw_response"] = t2["raw_response"][:2000]
            out.append(t2)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  ✅ COT trails -> {path}")
