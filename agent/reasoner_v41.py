"""V41: 三创新点整合 — 锚定状态① + 上下文手术刀② + 三档管家③

## 整合
① AnchorState: 构建锚定状态字典 (标的/意图/选项claim/比较维度) 注入 prompt
② ContextSurgeon: 倒U重排完整段落 (重要放首尾, 对抗中间迷失)
③ ThreeTierCompressor: 重要性打分 + 三档分层 (关键数据保留原貌段落/摘要/截断)

## vs V40 的关键修正
V40 失败: 关键数据压成单行碎片 → 模型看不全 → 保守少选 → 完全匹配0分 (退6.58分)
V41 修正: 创新点③"关键数据保留原貌" → 用 full_text 定位取 ±400字完整段落
  fin_a_001 验证: 4选项数据全在完整段落 (营收+3.46%/净利-18.97%/现金流-55.69%/研发7.89%vs6.97%)
  真值ABC, V40答B(漏AC), V41应能答ABC

## 分域
- financial_reports: 三创新点全开 (有 DocumentCard, 数字型, V40验证过原貌有效)
- 其他域: 保持 V35 (不引入风险, 待 fin 域验证成功再扩)

## 不自动跑全量
先零 token 验证证据含完整原貌段落, 再小样本 (¥0.19) 看 Acc, 见 full-eval-cost-gate.
"""
import os
import json
from agent.reasoner_v35 import ReasoningAgentV35
from agent.memory_compressor_v41 import ThreeTierCompressor
from agent.context_surgeon_v41 import u_shaped_reorder_segments, render_evidence
from agent.anchor_state_v41 import AnchorState
from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI
from agent.postprocessor import extract_answer_from_response
from agent.config import RESULTS_DIR


V41_MAX_EVIDENCE = {
    "tf": 8000,
    "mcq": 10000,
    "multi": 14000,  # 比V40的10K大, 留足原貌段落空间 (防少选)
}


class ReasoningAgentV41(ReasoningAgentV35):
    """V41: 三创新点整合 (fin域)."""

    def __init__(self, qwen, doc_index, vector_indexer=None, token_budget=5_000_000, model="qwen3.6-plus"):
        super().__init__(qwen, doc_index, vector_indexer, token_budget, model)
        self.compressor = ThreeTierCompressor()
        self.anchor = AnchorState()

    def answer_question(self, question: dict) -> dict:
        domain = question.get("domain", "")
        # 仅 fin 域走 V41 三创新点
        if domain == "financial_reports":
            return self._answer_v41(question)
        return ReasoningAgentV35.answer_question(self, question)

    def _answer_v41(self, question: dict) -> dict:
        """fin域: 锚定状态① + 三档压缩③ + 倒U重排② + V20 prompt."""
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        max_chars = V41_MAX_EVIDENCE.get(answer_format, 10000)

        # ① 锚定状态机: 构建状态字典
        _, state_text = self.anchor.build(question, self.doc_index)

        # ③ 三档压缩: 关键数据保留原貌段落 + 摘要
        segments, cstats = self.compressor.compress(question, self.doc_index, max_chars=max_chars)

        # ② 上下文手术刀: 倒U重排完整段落
        reordered = u_shaped_reorder_segments(segments)
        evidence = render_evidence(reordered)

        # 组装 prompt: 锚定状态 + 证据 (状态在前, 帮模型先锚定)
        full_evidence = state_text + "\n## 文档证据\n" + evidence

        # 题型分流 prompt (复用 V20)
        if answer_format == "tf":
            prompt_tpl = PROMPT_TF
        elif answer_format == "mcq":
            prompt_tpl = PROMPT_MCQ
        else:
            prompt_tpl = PROMPT_MULTI

        prompt = prompt_tpl.format(
            evidence=full_evidence, question=q_text,
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
            "evidence_chars": len(full_evidence),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": False,
            "raw_response": raw_response,
            "strategy": "v41_three_innovations",
            "compress_stat": cstats,
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(full_evidence),
            "total_doc_chars": total_doc_chars,
        }

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v41.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = []
        for t in self.cot_trails:
            t2 = dict(t)
            if "raw_response" in t2:
                t2["raw_response"] = t2["raw_response"][:1500]
            out.append(t2)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  V41 COT trails -> {path}")
