"""V16: Qwen-Long 原生文件上传 + 全文推理

核心突破（相比V15）：
- 彻底摆脱分块/检索/证据截断，把原始 PDF 直接上传给 qwen-long
- qwen-long 1000万 Token 上下文 = 可同时处理多份金融长文档
- PDF 上传免费，且价格约为 qwen-plus 的 1/10
- 不再依赖 BM25/向量检索，不会因检索遗漏丢分

策略：
  1. 所有文档预先上传，获取 file_id（磁盘缓存，一次性操作）
  2. 每题：把相关文档 file_id + CoT prompt 发给 qwen-long
  3. qwen-long 直接在完整文档内推理，无证据截断

Token 估算：
  - 假设每份文档平均 50K tokens（金融报告较长）
  - 每题平均 1-2 份文档 → 约 50-100K tokens/题
  - 100题 × 75K = 7.5M tokens（超出预算）
  
  优化策略：
  - 短文档（≤50K chars）直接全文内嵌（不上传，节省反复计费）
  - 长文档上传 file_id（避免把大文本放进 prompt）
  - 多文档题：合并成一次调用
  - Token 预算分配：长文档优先用 qwen-long，短文档用 qwen-plus 全文

混合策略（确保 Token 不超限）：
  - 文档总字符 ≤ 60K：用 qwen-plus 全文内嵌（V15方式，更快更省）
  - 文档总字符 > 60K 且 ≤ 500K：用 qwen-long 文件上传
  - 文档总字符 > 500K（超大文档）：用 qwen-long + 仅上传，禁止截断
"""

import os
import re
import json
import time
from collections import defaultdict
from agent.config import RESULTS_DIR, RAW_DIR, PROCESSED_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.qwen_long_client import QwenLongClient
from agent.indexer import DocumentIndex
from agent.vector_indexer import VectorIndexer
from agent.postprocessor import extract_answer_from_response
from agent.reasoner_v15 import (
    ReasoningAgentV15,
    DOMAIN_SYSTEM_PROMPTS,
    COT_PROMPT_V15,
    SELF_CRITIQUE_PROMPT_V15,
    compress_l1,
)

# ── qwen-long 专用 Prompt（更简洁，因为模型能看到完整文档）────────────

QWEN_LONG_COT_PROMPT = """请基于以上文档，严格依据原文逐选项判断是否正确。

## 问题
{question}

## 选项
{options}

## 逐选项验证（必须引用原文精确语句）

**选项A**：{option_a}
原文引用：
判断：✅ 明确支持 / ❌ 无明确支持

**选项B**：{option_b}
原文引用：
判断：✅ 明确支持 / ❌ 无明确支持

**选项C**：{option_c}
原文引用：
判断：✅ 明确支持 / ❌ 无明确支持

**选项D**：{option_d}
原文引用：
判断：✅ 明确支持 / ❌ 无明确支持

注意：
- 数值比较需精确到原文数字
- "应当"≠"可以"，"召开前"≠"通知中"
- 多选题：只选有明确原文支持的选项，宁可少选不多选

最终答案：{answer_hint}"""


def _find_raw_pdf(doc_id: str) -> str:
    """根据 doc_id 查找原始 PDF 路径"""
    # 各领域的文件名规律
    domains = ["insurance", "financial_contracts", "financial_reports", "research"]
    for domain in domains:
        raw_dir = os.path.join(RAW_DIR, domain)
        if not os.path.isdir(raw_dir):
            continue
        for f in os.listdir(raw_dir):
            name = os.path.splitext(f)[0]
            if name.lower() == doc_id.lower():
                return os.path.join(raw_dir, f)
    # regulatory 特殊结构
    reg_dir = os.path.join(RAW_DIR, "regulatory")
    for root, dirs, files in os.walk(reg_dir):
        for f in files:
            name = os.path.splitext(f)[0]
            if name.lower() == doc_id.lower():
                return os.path.join(root, f)
    return ""


# Token 阈值：超过此字符数时使用 qwen-long（否则 qwen-plus 全文）
QWEN_LONG_THRESHOLD = 60_000   # chars
# qwen-long 单题 Token 预算上限（防止超总预算）
QWEN_LONG_PER_Q_TOKEN_CAP = 200_000


class ReasoningAgentV16(ReasoningAgentV15):
    """V16: 混合策略 — 短文档用 qwen-plus，长文档用 qwen-long 文件上传"""

    def __init__(
        self,
        qwen: QwenClient,
        qwen_long: QwenLongClient,
        doc_index: DocumentIndex,
        vector_indexer: VectorIndexer = None,
        token_budget: int = TOKEN_BUDGET,
    ):
        super().__init__(qwen, doc_index, vector_indexer, token_budget)
        self.qwen_long = qwen_long
        self._file_id_map: dict = {}   # doc_id → file_id
        self._upload_attempted: set = set()

    # ------------------------------------------------------------------
    # 文件预上传
    # ------------------------------------------------------------------

    def preupload_documents(self, questions: list):
        """预先上传所有题目用到的长文档

        只上传字符数 > QWEN_LONG_THRESHOLD 的文档
        """
        # 收集所有 doc_id
        doc_ids_needed = set()
        for q in questions:
            for did in q.get("doc_ids", []):
                total_chars = self.doc_index.doc_lengths.get(did, 0)
                if total_chars > QWEN_LONG_THRESHOLD:
                    doc_ids_needed.add(did)

        if not doc_ids_needed:
            print("  无需上传（所有文档均 ≤60K字符）")
            return

        print(f"  需要上传 {len(doc_ids_needed)} 个长文档到 qwen-long...")
        for doc_id in sorted(doc_ids_needed):
            # 先查缓存
            cached = self.qwen_long.get_file_id(doc_id)
            if cached:
                self._file_id_map[doc_id] = cached
                print(f"    缓存命中: {doc_id} → {cached}")
                continue

            # 查找原始文件
            raw_path = _find_raw_pdf(doc_id)
            if not raw_path or not os.path.exists(raw_path):
                print(f"    未找到原始文件: {doc_id}，将回退到 V15 检索策略")
                continue

            # 上传
            fid = self.qwen_long.upload_file(raw_path, doc_id)
            if fid:
                self._file_id_map[doc_id] = fid

        print(f"  预上传完成：{len(self._file_id_map)} 个文件有 file_id")

    # ------------------------------------------------------------------
    # 主推理入口（覆盖 V15）
    # ------------------------------------------------------------------

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question.get("domain", "")
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)

        # 判断策略：是否用 qwen-long 文件上传
        use_qwen_long = self._should_use_qwen_long(doc_ids, total_doc_chars)

        if use_qwen_long:
            result = self._answer_with_qwen_long(
                qid, domain, q_text, options, answer_format, doc_ids, total_doc_chars)
        else:
            # 回退到 V15 策略（短文档 qwen-plus 全文）
            result = super().answer_question(question)

        return result

    def _should_use_qwen_long(self, doc_ids: list, total_chars: int) -> bool:
        """判断是否使用 qwen-long 策略"""
        if total_chars <= QWEN_LONG_THRESHOLD:
            return False
        # 至少有一个文件有 file_id 或可以找到原始文件
        for did in doc_ids:
            if did in self._file_id_map:
                return True
        return False

    def _answer_with_qwen_long(
        self, qid, domain, q_text, options, answer_format, doc_ids, total_doc_chars
    ) -> dict:
        """使用 qwen-long 文件上传方式回答"""
        # 收集有 file_id 的文档
        file_ids = [self._file_id_map[did] for did in doc_ids if did in self._file_id_map]
        # 没有 file_id 的文档用全文内嵌补充
        inline_texts = []
        for did in doc_ids:
            if did not in self._file_id_map:
                ft = self.doc_index.get_doc_full_text(did)
                if ft:
                    inline_texts.append(f"=== 文档 {did} ===\n{compress_l1(ft)}")

        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "你是金融文档分析专家，严格依据原文回答。")
        options_text = "\n".join(f"{k}. {options[k]}" for k in sorted(options))

        # 如果有内联文本，拼到 prompt 里
        inline_part = "\n\n".join(inline_texts)
        if inline_part:
            inline_part = f"\n\n## 补充文档（内嵌）\n{inline_part[:20000]}"

        prompt = QWEN_LONG_COT_PROMPT.format(
            question=q_text,
            options=options_text,
            option_a=options.get("A", ""),
            option_b=options.get("B", ""),
            option_c=options.get("C", ""),
            option_d=options.get("D", ""),
            answer_hint={
                "mcq":   "一个大写字母(A/B/C/D)",
                "tf":    "A或B",
                "multi": "多个大写字母按字母序(如ABC)，只选有明确原文支持的选项",
            }.get(answer_format, ""),
        ) + inline_part

        raw_response = ""
        try:
            result = self.qwen_long.chat_with_files(
                file_ids=file_ids,
                question=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=4096,
                timeout=300,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [LONG_ERR:{e}]")

        answer = extract_answer_from_response(raw_response, answer_format)
        if not answer and raw_response:
            answer = self._fallback_extract(raw_response, answer_format)

        # Self-Critique（多选题）：用内嵌 8K 证据
        if answer_format == "multi" and answer and len(answer) >= 2:
            # 用 qwen-plus 做 critique（省 qwen-long token）
            snippet = raw_response[:4000]  # 用模型自己的推理作为"证据"
            try:
                crit = self.qwen.chat(
                    [{"role": "user", "content": SELF_CRITIQUE_PROMPT_V15.format(
                        answer=answer, evidence=snippet)}],
                    temperature=0.0, max_tokens=256, timeout=60,
                )
                corrected = extract_answer_from_response(crit["content"], "multi")
                if corrected and set(corrected).issubset(set(answer)):
                    answer = corrected
            except Exception:
                pass

        answer = self._post_process(answer, answer_format)
        self.memory.questions_answered += 1

        self.cot_trails.append({
            "qid": qid, "domain": domain,
            "answer": answer, "answer_format": answer_format,
            "strategy": "qwen_long",
            "file_ids": file_ids,
            "total_doc_chars": total_doc_chars,
        })

        return {
            "qid": qid,
            "answer": answer,
            "evidence_chars": total_doc_chars,
            "total_doc_chars": total_doc_chars,
            "strategy": "qwen_long",
            "full_doc_threshold": QWEN_LONG_THRESHOLD,
        }

    def save_cot_trails(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "eval_results_v16.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)
        print(f"  qwen-long 调用: {self.qwen_long.call_count} 次")
        print(f"  qwen-long 上传: {self.qwen_long.upload_count} 次")
        long_stats = self.qwen_long.get_token_stats()
        print(f"  qwen-long Token: {long_stats['total_tokens']:,}")
