"""V15: V13 的最小增量优化（已验证安全的改动）

实验原则：
- 只改 V13 中已被证明有问题的地方，不改已证明有效的机制
- 每次改动必须有历史数据支撑，不靠猜测

V13 已证明的有效机制（保持不变）：
  - FULL_DOC_THRESHOLD = 50000
  - 融合证据池 RRF 3路（不分选项）
  - EVIDENCE_LIMITS = {mcq:20K, tf:15K, multi:25K}
  - V9 COT_PROMPT（逐选项验证，已产生 A=39 均衡分布）
  - 领域专用 System Prompt

V15 新增（低风险，有理论依据）：
  1. L1 增强压缩：仅对全文模式下的文档做额外的页眉页脚清理
     - 理论依据：去除页码/保密标记不影响内容，让同等 Token 携带更多有效信息
     - 风险：极低，只去除明确的格式噪声
     - 预期收益：全文模式下证据质量提升

  2. 空答案补救：mcq/tf 答案为空时从响应末尾反向扫描
     - 理论依据：模型有时在非标准格式下输出答案（如"答案是B"而不是"最终答案：B"）
     - 风险：极低，只作兜底
     - 预期收益：减少空答案（当前 V13 偶有空答案）

  3. _post_process 去默认 A：mcq/tf 无法提取时返回空串而非 "A"
     - 理论依据：盲猜 A 正确率约 25%，不如返回空（计错但不浪费 Token）
     - 注意：如果空答案过多会降低有效答案数，需权衡

放弃的改动（有风险或已证明有害）：
  - Stage1 Locate 调用：增加了 A 偏置（类比 V10.1），禁用
  - Self-Critique（带证据）：V13 的 self-critique 已证明稳健，V15 版本反而更差
  - 题型差异化阈值：引入不一致的文档处理路径，禁用
  - 增大 EVIDENCE_LIMITS：Token 增加但无提升证据

Token 预算：
  - V13 实际：1.85M，TokenScore=0.630
  - V15 预期：≈1.85M（与 V13 完全一致），TokenScore≈0.630
  - 无额外 API 调用

若 V15 得分 ≤ V13，说明 L1 压缩或空答案补救有副作用，需回退到 V13。
"""
from agent.reasoner_v13 import (
    ReasoningAgentV13,
    DOMAIN_SYSTEM_PROMPTS,
    COT_PROMPT,
    CLAUSE_PATTERN,
    MONEY_PATTERN,
    PERCENT_PATTERN,
    DATE_PATTERN,
    extract_query_keywords,
    keyword_match_score,
    compress_whitespace,
    AgentMemory,
)
from agent.postprocessor import extract_answer_from_response
import re
import os
import json
from agent.config import RESULTS_DIR

# ── 增强 L1 压缩（仅用于全文模式，不影响检索模式的证据块）─────────────
_HEADER_FOOTER_RE = [
    re.compile(r'^第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$', re.MULTILINE),
    re.compile(r'^\s*[-—–]\s*\d+\s*[-—–]\s*$', re.MULTILINE),
    re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE),  # 仅1-3位纯数字独立行（页码），不影响金融数字
]


def _compress_l1_fulltext(text: str) -> str:
    """全文模式专用 L1 压缩：去除页眉/页脚 + compress_whitespace"""
    for pat in _HEADER_FOOTER_RE:
        text = pat.sub('', text)
    return compress_whitespace(text)


class ReasoningAgentV15(ReasoningAgentV13):
    """V15: V13 的最小安全增量

    改动1：全文模式下用 _compress_l1_fulltext 替代 compress_whitespace
    改动2：空答案补救扫描
    改动3：_post_process 去默认 A（mcq/tf 无法提取时返回空串）
    """

    def answer_question(self, question: dict) -> dict:
        """完全复用 V13 逻辑，仅在全文模式替换压缩函数"""
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        card_hints = self.memory.get_card_match_hints(q_text, options, doc_ids)

        total_doc_chars = sum(
            self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        is_full_doc = total_doc_chars <= self.FULL_DOC_THRESHOLD

        # ── 证据收集（改动1：全文模式用增强压缩）────────────────────────
        if is_full_doc:
            evidence_text = ""
            for doc_id in doc_ids:
                ft = self.doc_index.get_doc_full_text(doc_id)
                if ft:
                    compressed = _compress_l1_fulltext(ft)  # V15改动：增强压缩
                    evidence_text += f"\n=== 文档 {doc_id} (全文) ===\n{compressed}\n"
        else:
            # 检索模式完全复用 V13（不改动）
            evidence_text = self._retrieve_merged_evidence(
                q_text, options, doc_ids, card_hints, answer_format)

        # ── 推理（完全复用 V13）─────────────────────────────────────────
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")
        prompt = COT_PROMPT.format(
            evidence=evidence_text,
            question=q_text,
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
            option_a=options.get("A", ""),
            option_b=options.get("B", ""),
            option_c=options.get("C", ""),
            option_d=options.get("D", ""),
            answer_hint={
                "mcq": "一个大写字母(A/B/C/D)",
                "tf": "A或B",
                "multi": "多个大写字母按字母序(如ABC)，只包含有明确原文支持的选项",
            }.get(answer_format, ""),
        )

        try:
            result = self.qwen.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, max_tokens=4096, timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""

        answer = extract_answer_from_response(raw_response, answer_format)

        # 改动2：空答案补救（从响应末尾反向扫描，仅兜底）
        if not answer and raw_response:
            answer = self._v15_fallback_extract(raw_response, answer_format)

        # Self-Critique 完全复用 V13（仅多选题，已证明稳定）
        if answer_format == "multi" and answer and len(answer) >= 2:
            try:
                from agent.reasoner_v13 import SELF_CRITIQUE_PROMPT
                critique_result = self.qwen.chat(
                    [{"role": "user", "content": SELF_CRITIQUE_PROMPT.format(answer=answer)}],
                    temperature=0.0, max_tokens=256, timeout=60,
                )
                corrected = extract_answer_from_response(
                    critique_result["content"], "multi")
                if corrected and set(corrected).issubset(set(answer)):
                    answer = corrected
            except Exception:
                pass

        answer = self._post_process(answer, answer_format)
        self.memory.questions_answered += 1

        self.cot_trails.append({
            "qid": qid, "domain": domain,
            "answer": answer, "answer_format": answer_format,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": is_full_doc,
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
            "full_doc_threshold": self.FULL_DOC_THRESHOLD,
            "is_full_doc": is_full_doc,
        }

    def _v15_fallback_extract(self, text: str, answer_format: str) -> str:
        """空答案兜底：从响应末尾反向扫描合法字母"""
        valid = set("ABCD")
        if answer_format == "mcq":
            for c in reversed(text):
                if c in valid:
                    return c
        elif answer_format == "tf":
            for c in reversed(text):
                if c in ("A", "B"):
                    return c
        elif answer_format == "multi":
            letters = sorted(set(c for c in text if c in valid))
            if letters:
                return "".join(letters[:3])
        return ""

    def _post_process(self, answer: str, answer_format: str) -> str:
        """改动3：mcq/tf 无答案时返回空串，不默认回退 A"""
        if not answer:
            return ""
        valid_letters = set("ABCD")
        if answer_format == "mcq":
            for c in answer:
                if c in valid_letters:
                    return c
            return ""   # V15改动：不默认A
        elif answer_format == "tf":
            if "A" in answer:
                return "A"
            if "B" in answer:
                return "B"
            return ""   # V15改动：不默认A
        elif answer_format == "multi":
            letters = sorted(set(c for c in answer if c in valid_letters))
            if len(letters) > 3:
                letters = letters[:3]
            return "".join(letters) if letters else ""
        return answer

    def save_cot_trails(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "eval_results_v15.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)
        card_path = os.path.join(RESULTS_DIR, "document_cards_v15.json")
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(self.memory.cards, f, ensure_ascii=False, indent=2)
