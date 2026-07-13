"""V35: V31 + multi 证据重构（上限60K + head在前锚词追加）

SOP V32 方案：保 Acc 降 Token
- 只改 multi 证据构造，不改 tf/mcq、不改 prompt、不改后处理
- fc/fin 域：证据上限 90K→60K，head 在前(40%) + 锚词追加(60%)
- research/insurance 域：保持 V22 的 45K head（不降，避免丢失后部内容）
- 证据顺序保持 V22 的 head 在前 + 锚词追加（不反转，避免 V13.1 段排序退分）

V31 multi 证据问题：
- V22: head(35K) + 锚词追加 → per_doc=30K时锚词空间=0 → 丢失关键信息
- V35: head(40%) + 锚词追加(60%) → 锚词一定有空间

证据包结构（SOP 4.3）：
1. 文档头部摘要（head 在前，模型更关注开头）
2. 锚词章节追加在后（确保关键条款不丢）
3. 不反转顺序（V13.1 教训：段顺序改变可能退分）
"""
from agent.reasoner_v31 import ReasoningAgentV31
from agent.reasoner_v22 import build_evidence_v22
from agent.reasoner_v21 import V21_FC_ANCHORS
from agent.reasoner_v20 import (
    _take_head, _locate_section,
    DOMAIN_SYSTEM, PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI,
    extract_evidence_regulatory,
)
from agent.postprocessor import extract_answer_from_response
import os, json
from agent.config import RESULTS_DIR


# ============ V35 证据提取器（head 在前 + 锚词追加）============

def extract_evidence_fc_v35(text: str, max_chars: int = 30000) -> str:
    """V35 金融合同：head 在前(40%) + 锚词追加在后(60%)

    顺序与 V22 一致（head 在前），仅缩小 head 让锚词有空间。
    """
    # head 在前（占约 40%，至少 8K）
    head_budget = max(8000, int(max_chars * 0.4))
    head = _take_head(text, head_budget)
    parts = [head]
    seen = set()

    # 锚词章节追加在后（V21 的 26 个锚词）
    anchor_ctx = min(2000, max_chars // 10)
    for anchor in V21_FC_ANCHORS:
        if sum(len(p) for p in parts) >= max_chars:
            break
        sec = _locate_section(text, [anchor], ctx_chars=anchor_ctx)
        if sec:
            key = sec[:50]
            if key in seen or key in head:
                continue
            seen.add(key)
            parts.append(f"[{anchor}]\n{sec}")

    result = "\n\n".join(parts)
    return result[:max_chars]


def extract_evidence_fin_v35(text: str, max_chars: int = 30000) -> str:
    """V35 财报：head 在前(40%) + 关键章节追加在后(60%)

    顺序与 V22 一致（head 在前），仅缩小 head 让章节有空间。
    """
    # head 在前
    head_budget = max(8000, int(max_chars * 0.4))
    head = _take_head(text, head_budget)
    parts = [head]
    seen = set()

    # 1. 主要会计数据（最高优先级）
    for anchor in ["主要会计数据", "主要财务指标"]:
        if sum(len(p) for p in parts) >= max_chars:
            break
        sec = _locate_section(text, [anchor], ctx_chars=min(6000, max_chars // 5))
        if sec:
            key = sec[:50]
            if key not in seen and key not in head:
                seen.add(key)
                parts.append(f"[主要会计数据]\n{sec}")
                break

    # 2. 现金流/分红
    for anchor in ["现金流量净额", "现金分红", "利润分配", "派发现金"]:
        if sum(len(p) for p in parts) >= max_chars:
            break
        sec = _locate_section(text, [anchor], ctx_chars=min(4000, max_chars // 8))
        if sec:
            key = sec[:50]
            if key not in seen and key not in head:
                seen.add(key)
                parts.append(f"[现金流/分红]\n{sec}")
                break

    # 3. 研发/回购
    for anchor in ["研发投入", "研发费用", "股份回购", "回购"]:
        if sum(len(p) for p in parts) >= max_chars:
            break
        sec = _locate_section(text, [anchor], ctx_chars=min(2000, max_chars // 10))
        if sec:
            key = sec[:50]
            if key not in seen and key not in head:
                seen.add(key)
                parts.append(f"[{anchor}]\n{sec}")

    result = "\n\n".join(parts)
    return result[:max_chars]


def extract_evidence_simple_v35(text: str, max_chars: int = 45000) -> str:
    """V35 research/insurance：纯 head（与 V22 相同，不缩小）

    research/insurance 域没有锚词机制，纯靠 head。
    build_evidence_v35 中 per_doc 固定 45K，保持与 V22 一致。
    """
    return _take_head(text, max_chars)


V35_DOMAIN_EXTRACTORS = {
    "financial_reports": extract_evidence_fin_v35,
    "financial_contracts": extract_evidence_fc_v35,
    "research": extract_evidence_simple_v35,
    "insurance": extract_evidence_simple_v35,
    "regulatory": extract_evidence_regulatory,
}


def build_evidence_v35(doc_index, domain, doc_ids, max_evidence):
    """V35 证据组装

    - fc/fin: per_doc = max_evidence / n_docs（降到 60K）
    - research/insurance: per_doc = 45K（保持 V22，不降）
    - regulatory: 走 V20 extract_evidence_regulatory
    """
    extractor = V35_DOMAIN_EXTRACTORS.get(domain, extract_evidence_regulatory)
    n_docs = max(1, len(doc_ids))

    # research/insurance 保持 V22 的 45K/doc（不降，避免丢失后部内容）
    if domain in ("research", "insurance"):
        per_doc = 45000
    else:
        per_doc = max_evidence // n_docs

    evidence = ""
    for did in doc_ids:
        t = doc_index.get_doc_full_text(did) or ""
        if not t:
            continue
        seg = extractor(t, max_chars=per_doc)
        evidence += f"\n=== 文档 {did} ===\n{seg}\n"
    return evidence


class ReasoningAgentV35(ReasoningAgentV31):
    """V35: V31 + multi 证据重构（上限60K + head在前锚词追加）

    只改 multi 证据构造：
    - MULTI_MAX_EVIDENCE: 90K → 60K（仅对 fc/fin 域生效）
    - research/insurance 保持 45K/doc（不降）
    - 证据顺序：head 在前 + 锚词追加（与 V22 一致，不反转）
    - tf/mcq: 完全复用 V31（V30 精炼）
    - prompt: 完全复用 V20
    - 后处理: 完全复用 V20（保留 fallback A）
    """

    MULTI_MAX_EVIDENCE = 60000

    def answer_question(self, question: dict) -> dict:
        answer_format = question.get("answer_format", "mcq")

        if answer_format == "multi":
            return self._answer_multi_v35(question)
        else:
            # tf/mcq 完全复用 V31(=V30 精炼)
            return ReasoningAgentV31.answer_question(self, question)

    def _answer_multi_v35(self, question: dict) -> dict:
        """V35 multi 路径：head在前+锚词追加证据 + V20 prompt"""
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)

        evidence = build_evidence_v35(
            self.doc_index, domain, doc_ids, self.MULTI_MAX_EVIDENCE)

        prompt = PROMPT_MULTI.format(
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

        answer = extract_answer_from_response(raw_response, "multi")
        answer = self._post_process(answer, "multi")

        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": answer,
            "answer_format": "multi",
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": False,
            "raw_response": raw_response,
            "strategy": "v35_multi",
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
        }

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v35.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = [dict(t, raw_response=t.get("raw_response", "")[:1500]) for t in self.cot_trails]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  V35 COT trails -> {path}")
