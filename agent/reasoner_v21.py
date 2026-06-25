"""V21: V20 的微改 — 仅扩展 financial_contracts 域关键词

V20 = 43.81 是新基线. V21 严格单变量改动:
- ✅ 仅改 `extract_evidence_financial_contract` 锚词列表 (从 5 个扩到 26 个)
- ❌ 不改其他域抽取
- ❌ 不改 prompt 模板
- ❌ 不改后处理
- ❌ 不改 reasoner 主流程

V20 fc 失败案例: fc_a_014 选项 B "5 日/10 日 报告出具" — 关键条款在 text08 @ 305K
V20 锚词不含 "资产减值"/"业绩承诺"/"通知乙方", 漏召回 → 答错

V21 加 22 个金融合同常见锚词 (业绩补偿/资产减值/通知乙方/股份回购/票面利率等)
离线测试: text08 "5 日" V20=0 → V21=1, "补偿" V20=17 → V21=107

预期: fc 域 20 题中, V20 推理 raw 多次"未出现/未提及"的题 (fc_a_005/006/007/008/009/012/013/014/015) 可能改对
其他 80 题答案不变 (相同 prompt 输入)
"""
from agent.reasoner_v20 import (
    ReasoningAgentV20,
    extract_evidence_financial_report,
    extract_evidence_research,
    extract_evidence_insurance,
    extract_evidence_regulatory,
    _take_head, _locate_section,
    DOMAIN_EXTRACTORS, DOMAIN_SYSTEM,
    PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI, COMMON_HEADER,
    build_evidence,
)


# V21 扩展 fc 锚词 (按金融合同常见条款类型分组)
V21_FC_ANCHORS = [
    # 转股 / 赎回 / 回售
    "转股价格的修正", "向下修正", "转股价格", "赎回条款", "回售条款",
    # 违约
    "违约事件", "违约利息", "违约责任",
    # 业绩补偿 / 资产减值
    "资产减值", "减值测试报告", "业绩补偿", "业绩承诺",
    "补偿协议", "补偿义务", "现金补偿",
    # 通知 / 期限
    "通知乙方", "通知期限", "股份回购",
    # 兑付 / 利率
    "兑付日", "兑付期", "本金", "票面利率",
    # 锁定
    "认购股份", "解锁股份",
    # 评级 / 募集
    "信用评级", "主体评级", "债项评级",
    "募集资金", "发行规模",
]


def extract_evidence_financial_contract_v21(text: str, max_chars: int = 50000) -> str:
    """V21: 扩展锚词版"""
    head = _take_head(text, 35000)
    seen = set()
    for anchor in V21_FC_ANCHORS:
        sec = _locate_section(text, [anchor], ctx_chars=3500)
        if sec:
            key = sec[:50]
            if key in seen or key in head:
                continue
            seen.add(key)
            head += f"\n[{anchor}]\n" + sec
            if len(head) >= max_chars:
                break
    return head[:max_chars]


# 替换 extractor (仅 fc 域)
V21_DOMAIN_EXTRACTORS = dict(DOMAIN_EXTRACTORS)
V21_DOMAIN_EXTRACTORS["financial_contracts"] = extract_evidence_financial_contract_v21


def build_evidence_v21(doc_index, domain, doc_ids, n_docs, max_evidence):
    """按域提取证据 (V21 用扩展 fc 锚词)"""
    extractor = V21_DOMAIN_EXTRACTORS.get(domain, extract_evidence_regulatory)
    per_doc = max_evidence // max(1, n_docs)
    evidence = ""
    for did in doc_ids:
        t = doc_index.get_doc_full_text(did) or ""
        if not t: continue
        seg = extractor(t, max_chars=per_doc)
        evidence += f"\n=== 文档 {did} ===\n{seg}\n"
    return evidence


class ReasoningAgentV21(ReasoningAgentV20):
    """V21: V20 + fc 锚词扩展"""

    def answer_question(self, question: dict) -> dict:
        # 完整复用 V20 流程, 仅证据构造换成 V21
        from agent.postprocessor import extract_answer_from_response
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        max_ev = self.MAX_EVIDENCE.get(answer_format, 80000)

        # V21 证据构造
        evidence = build_evidence_v21(self.doc_index, domain, doc_ids, len(doc_ids), max_ev)

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
        path = path or os.path.join(RESULTS_DIR, "eval_results_v21.json")
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
