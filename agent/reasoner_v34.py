"""V34: V31 + enable_thinking=False

V31 使用 qwen3.6-plus 但没有关闭 thinking 模式，
thinking chain 消耗了大量 token（估计 30-50% of 3.19M）。

V34 = V31 的所有推理路径，只加 enable_thinking=False。
零 Acc 风险，预计 Token 降至 1.8-2.0M，TS≈0.62-0.64。
Score = 60% × (0.7 + 0.3 × 0.63) = 60% × 0.889 ≈ 53.3
"""
import os
import json
from agent.reasoner_v31 import ReasoningAgentV31
from agent.config import RESULTS_DIR


class ReasoningAgentV34(ReasoningAgentV31):
    """V34 = V31 + enable_thinking=False on all LLM calls"""

    def answer_question(self, question: dict) -> dict:
        answer_format = question.get("answer_format", "mcq")
        if answer_format == "multi":
            return self._answer_multi_v34(question)
        else:
            return self._answer_tfmcq_v34(question)

    def _answer_multi_v34(self, question: dict) -> dict:
        """V22 full-text path, thinking disabled"""
        from agent.reasoner_v22 import ReasoningAgentV22
        from agent.reasoner_v20 import DOMAIN_SYSTEM
        from agent.postprocessor import extract_answer_from_response

        qid = question["qid"]
        domain = question["domain"]
        answer_format = question.get("answer_format", "multi")
        system = DOMAIN_SYSTEM.get(domain, "")

        # Reuse V22's evidence building (all the retrieve/compress logic)
        # but call qwen with enable_thinking=False
        # We do this by temporarily patching, or just copy V22's logic inline.

        # Simplest: call parent V22 path but intercept the chat call.
        # V22.answer_question calls self.qwen.chat(...) — since self is V34,
        # we override qwen.chat to always pass enable_thinking=False.
        original_chat = self.qwen.chat
        def chat_no_think(messages, temperature=0.1, max_tokens=4096,
                          timeout=180, enable_thinking=True):
            return original_chat(messages, temperature=temperature,
                                 max_tokens=max_tokens, timeout=timeout,
                                 enable_thinking=False)
        self.qwen.chat = chat_no_think
        try:
            result = ReasoningAgentV31.answer_question(
                self, dict(question, answer_format="multi"))
        finally:
            self.qwen.chat = original_chat
        return result

    def _answer_tfmcq_v34(self, question: dict) -> dict:
        """V30 compact path, thinking disabled"""
        from agent.reasoner_v30 import ReasoningAgentV30

        original_chat = self.qwen.chat
        def chat_no_think(messages, temperature=0.1, max_tokens=4096,
                          timeout=180, enable_thinking=True):
            return original_chat(messages, temperature=temperature,
                                 max_tokens=max_tokens, timeout=timeout,
                                 enable_thinking=False)
        self.qwen.chat = chat_no_think
        try:
            result = ReasoningAgentV30.answer_question(self, question)
        finally:
            self.qwen.chat = original_chat
        return result

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v34.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = [dict(t, raw_response=t.get("raw_response", "")[:2000])
               for t in self.cot_trails]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  [OK] COT trails -> {path}")
