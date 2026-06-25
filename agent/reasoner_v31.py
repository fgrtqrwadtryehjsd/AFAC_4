"""V31: 动态证据分配 — multi→V22完整提取, tf/mcq→V30精炼提取

V22: 4.25M token, TokenScore=0.149, Acc=60%, Score=44.69
V30: 1.03M token, TokenScore=0.795, Acc=37%, Score=34.72  (multi 崩溃)
V31: ~3.3M token, TokenScore≈0.345, 预期 Acc≈60% → Score≈48.2

根因: multi 题需验证 4 个选项, V30 的 8-14K 证据覆盖不足
解法: tf/mcq(35题) 用 V30 精炼; multi(65题) 用 V22 完整提取(90K)
"""
import os
import json
from agent.reasoner_v22 import ReasoningAgentV22
from agent.reasoner_v30 import ReasoningAgentV30, build_evidence_v30
from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_TF, PROMPT_MCQ, PROMPT_MULTI
from agent.postprocessor import extract_answer_from_response
from agent.config import RESULTS_DIR


class ReasoningAgentV31(ReasoningAgentV22):
    """V31: tf/mcq → V30 精炼(8-14K); multi → V22 完整(90K)"""

    def answer_question(self, question: dict) -> dict:
        answer_format = question.get("answer_format", "mcq")
        if answer_format == "multi":
            return ReasoningAgentV22.answer_question(self, question)
        else:
            return ReasoningAgentV30.answer_question(self, question)

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v31.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = [dict(t, raw_response=t.get("raw_response", "")[:2000]) for t in self.cot_trails]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  [OK] COT trails -> {path}")
