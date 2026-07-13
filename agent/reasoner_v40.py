"""V40: V35 + financial_reports 域用 V40a 事实包 (分域分流)

## 设计 (用户决策: 仅 fin 域用事实包)
- financial_reports 域 (所有题型): V40a 事实包 (4K, 结构化事实+倒U重排)
  - 已离线验证: 79% 选项有数据支持, 4K 覆盖 V31 需 90K 的关键数字
  - 省 token + 数据更全 (V13 致命缺陷=关键数字 0% 覆盖, V40a 修复)
- 其他域 (insurance/contracts/regulatory/research): 完全保持 V35 (不引入风险)

## 继承链
ReasoningAgentV40 → ReasoningAgentV35 → ReasoningAgentV31 → ReasoningAgentV22/V30

## 单变量原则
仅改 fin 域证据构造, 不改 prompt / 后处理 / 模型参数 / 其他域.
fin 域: tf/mcq/multi 都走 V40a 事实包 (统一, 便于归因).

## 不自动跑全量
先小样本 (fin 域题) LLM 验证 Acc 是否真涨, 再决定全量 (见 full-eval-cost-gate).
"""
import os
import json
from agent.reasoner_v35 import ReasoningAgentV35
from agent.evidence_builder_v40 import FactCardStore, build_fact_pack
from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI
from agent.postprocessor import extract_answer_from_response
from agent.config import RESULTS_DIR


# fin 域事实包上限 (V40a 验证: 4K 够覆盖核心数据, 留余量到 8K)
V40_FIN_MAX_CHARS = {
    "tf": 6000,
    "mcq": 8000,
    "multi": 10000,
}


class ReasoningAgentV40(ReasoningAgentV35):
    """V40: V35 + fin 域 V40a 事实包."""

    def __init__(self, qwen, doc_index, vector_indexer=None, token_budget=5_000_000, model="qwen3.6-plus"):
        super().__init__(qwen, doc_index, vector_indexer, token_budget, model)
        self.fact_store = FactCardStore()

    def answer_question(self, question: dict) -> dict:
        domain = question.get("domain", "")
        # 仅 fin 域走 V40a 事实包
        if domain == "financial_reports":
            return self._answer_fin_v40(question)
        # 其他域完全保持 V35
        return ReasoningAgentV35.answer_question(self, question)

    def _answer_fin_v40(self, question: dict) -> dict:
        """fin 域: V40a 事实包 + V20 prompt (题型分流)."""
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        max_chars = V40_FIN_MAX_CHARS.get(answer_format, 8000)

        # V40a 事实包 (结构化事实 + 倒U重排)
        evidence, fstats = build_fact_pack(question, self.doc_index, self.fact_store,
                                           max_chars=max_chars)

        # 题型分流 prompt (复用 V20)
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
            "strategy": "v40_fin_factpack",
            "fact_stats": fstats,
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
        }

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v40.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = []
        for t in self.cot_trails:
            t2 = dict(t)
            if "raw_response" in t2:
                t2["raw_response"] = t2["raw_response"][:1500]
            out.append(t2)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  V40 COT trails -> {path}")
