"""V14: 平衡检索与自适应证据架构。

在 V13 融合证据池基础上引入：
1. 题型自适应全文阈值
2. 选项平衡检索与补充证据
3. 更稳健的后处理（去默认 A 偏置）
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
)
from agent.postprocessor import extract_answer_from_response


class ReasoningAgentV14(ReasoningAgentV13):
    """V14 在 V13 上做低风险增量优化。"""

    FULL_DOC_THRESHOLDS = {
        "tf": 120000,
        "mcq": 90000,
        "multi": 70000,
    }

    EVIDENCE_LIMITS = {
        "mcq": 24000,
        "tf": 18000,
        "multi": 32000,
    }

    OPTION_EVIDENCE_LIMITS = {
        "mcq": 2200,
        "tf": 1800,
        "multi": 2600,
    }

    ENABLE_SELF_CRITIQUE = False

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        card_hints = self.memory.get_card_match_hints(q_text, options, doc_ids)

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        full_doc_threshold = self.FULL_DOC_THRESHOLDS.get(answer_format, 80000)
        is_full_doc = total_doc_chars <= full_doc_threshold

        if is_full_doc:
            evidence_text = ""
            for doc_id in doc_ids:
                ft = self.doc_index.get_doc_full_text(doc_id)
                if ft:
                    evidence_text += f"\n=== 文档 {doc_id} (全文) ===\n{compress_whitespace(ft)}\n"
        else:
            evidence_text = self._retrieve_balanced_evidence(
                q_text, options, doc_ids, card_hints, answer_format
            )

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
                temperature=0.1,
                max_tokens=4096,
                timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""

        answer = extract_answer_from_response(raw_response, answer_format)

        if self.ENABLE_SELF_CRITIQUE and answer_format == "multi" and answer and len(answer) >= 2:
            try:
                critique = self.qwen.chat(
                    [
                        {
                            "role": "user",
                            "content": (
                                f"多选题 {answer} 的二次校验。"
                                "对每个已选选项，确认文档中是否有明确原文支持。"
                                "无明确支持则删除。不能添加新选项。\n最终答案："
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    timeout=60,
                )
                corrected = extract_answer_from_response(critique["content"], "multi")
                if corrected and set(corrected).issubset(set(answer)):
                    answer = corrected
            except Exception:
                pass

        answer = self._post_process(answer, answer_format)
        self.memory.questions_answered += 1

        self.cot_trails.append(
            {
                "qid": qid,
                "domain": domain,
                "answer": answer,
                "answer_format": answer_format,
                "evidence_chars": len(evidence_text),
                "total_doc_chars": total_doc_chars,
                "is_full_doc": is_full_doc,
                "full_doc_threshold": full_doc_threshold,
            }
        )

        return {
            "qid": qid,
            "answer": answer,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
            "full_doc_threshold": full_doc_threshold,
        }

    def _collect_option_metric_terms(self, q_text: str, options: dict) -> list:
        terms = set()
        query = q_text + " " + " ".join(options.values())
        for pattern in [MONEY_PATTERN, PERCENT_PATTERN, DATE_PATTERN]:
            for m in pattern.finditer(query):
                terms.add(m.group())
        return list(terms)

    def _retrieve_balanced_evidence(
        self,
        q_text: str,
        options: dict,
        doc_ids: list,
        card_hints: dict,
        answer_format: str,
    ) -> str:
        max_evidence = self.EVIDENCE_LIMITS.get(answer_format, 22000)
        per_option_cap = self.OPTION_EVIDENCE_LIMITS.get(answer_format, 2200)

        bm25_results = self.doc_index.search_bm25(q_text, top_k=22, doc_ids=doc_ids)

        vector_chunks = []
        if self.vector:
            q_results = self.vector.search_vector(q_text, top_k=18, doc_ids=doc_ids)
            opt_queries = [f"{q_text} {options[k]}" for k in sorted(options.keys())]
            opt_results = self.vector.search_vector_multi_query(opt_queries, top_k=18, doc_ids=doc_ids)
            fused_vector = self.vector.rrf_fuse(q_results, opt_results, top_n=30)
            for idx, _ in fused_vector:
                if idx < len(self.doc_index.chunks):
                    chunk = self.doc_index.chunks[idx]
                    if chunk.get("doc_id") in doc_ids:
                        vector_chunks.append(chunk)

        option_chunks = []
        for opt_key in sorted(options.keys()):
            opt_text = options.get(opt_key, "")
            if not opt_text:
                continue
            opt_query = f"{q_text} {opt_text}"

            opt_bm25 = self.doc_index.search_bm25(opt_query, top_k=6, doc_ids=doc_ids)
            for chunk in opt_bm25[:3]:
                option_chunks.append({**chunk, "chunk_type": f"option_{opt_key}"})

            if self.vector:
                opt_vec = self.vector.search_vector(opt_query, top_k=6, doc_ids=doc_ids)
                for idx, _ in opt_vec[:3]:
                    if idx < len(self.doc_index.chunks):
                        chunk = self.doc_index.chunks[idx]
                        if chunk.get("doc_id") in doc_ids:
                            option_chunks.append({**chunk, "chunk_type": f"option_{opt_key}"})

        query_keywords = extract_query_keywords(q_text, options)
        search_keywords = list(
            set(query_keywords + card_hints.get("entity_terms", []) + card_hints.get("key_term_hits", []))
        )
        all_doc_chunks = self.doc_index.get_chunks_by_doc_ids(doc_ids)
        kw_scored = []
        for chunk in all_doc_chunks:
            text = chunk.get("text", "")
            score = keyword_match_score(search_keywords, text)
            if score > 0:
                kw_scored.append((score, chunk))
        kw_scored.sort(key=lambda x: -x[0])
        kw_results = [chunk for _, chunk in kw_scored[:70]]

        clause_evidence = []
        for doc_id in doc_ids:
            for clause_ref in card_hints.get("clause_refs", []):
                clause_text = self.memory.locate_clause_text(doc_id, clause_ref, max_len=3000)
                if clause_text:
                    clause_evidence.append(
                        {
                            "doc_id": doc_id,
                            "text": clause_text,
                            "chunk_type": "clause_precise",
                            "clause_ref": clause_ref,
                        }
                    )

        metric_evidence = []
        metric_terms = list(set(card_hints.get("metric_contexts", []) + self._collect_option_metric_terms(q_text, options)))
        for doc_id in doc_ids:
            for metric_val in metric_terms:
                metric_ctx = self.memory.locate_metric_context(doc_id, metric_val)
                if metric_ctx:
                    metric_evidence.append(
                        {
                            "doc_id": doc_id,
                            "text": metric_ctx,
                            "chunk_type": "metric_precise",
                        }
                    )

        bm25_kw = self._rrf_fuse_chunks(
            bm25_results,
            kw_results,
            bm25_weight=1.0,
            kw_weight=1.5,
            top_n=45,
        )

        final_chunks = self._rrf_fuse_chunks_4way(
            clause_evidence + metric_evidence,
            bm25_kw,
            vector_chunks,
            option_chunks,
            top_n=80,
        )

        evidence_text = ""
        seen = set()
        option_buckets = {k: [] for k in sorted(options.keys())}

        for chunk in final_chunks:
            doc_id = chunk.get("doc_id", "")
            text = chunk.get("text", "")
            ref = chunk.get("clause_ref", "")
            ctype = chunk.get("chunk_type", "")
            text_key = text[:100]
            if text_key in seen:
                continue
            seen.add(text_key)

            label = f"文档 {doc_id}" + (f" [{ref}]" if ref else "")
            evidence_text += f"\n--- 来自{label} ---\n{text}\n"

            if ctype.startswith("option_"):
                opt = ctype.split("_", 1)[1]
                if opt in option_buckets:
                    option_buckets[opt].append(text)

            if len(evidence_text) > max_evidence:
                break

        for opt_key in sorted(options.keys()):
            if len(evidence_text) > max_evidence:
                break
            joined = "\n".join(option_buckets.get(opt_key, []))
            if not joined:
                continue
            snippet = joined[:per_option_cap].strip()
            if not snippet:
                continue
            evidence_text += f"\n=== 选项{opt_key}补充证据 ===\n{snippet}\n"
            if len(evidence_text) > max_evidence:
                break

        if len(evidence_text) > max_evidence:
            evidence_text = evidence_text[:max_evidence] + "\n[...证据已截断...]"

        return evidence_text

    def _rrf_fuse_chunks_4way(
        self,
        clause_results,
        bm25_kw_results,
        vector_chunks,
        option_chunks,
        top_n=80,
    ) -> list:
        chunk_scores = {}
        k = 60

        for rank, chunk in enumerate(clause_results):
            key = chunk.get("text", "")[:100]
            rrf = 2.0 / (k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf

        for rank, chunk in enumerate(bm25_kw_results):
            key = chunk.get("text", "")[:100]
            rrf = 1.0 / (k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf

        for rank, chunk in enumerate(vector_chunks):
            key = chunk.get("text", "")[:100]
            rrf = 1.2 / (k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf

        for rank, chunk in enumerate(option_chunks):
            key = chunk.get("text", "")[:100]
            rrf = 1.4 / (k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = {"score": 0, "chunk": chunk}
            chunk_scores[key]["score"] += rrf

        sorted_results = sorted(chunk_scores.values(), key=lambda x: -x["score"])
        return [item["chunk"] for item in sorted_results[:top_n]]

    def _post_process(self, answer: str, answer_format: str) -> str:
        if not answer:
            return ""

        valid_letters = set("ABCD")
        if answer_format == "mcq":
            for c in answer:
                if c in valid_letters:
                    return c
            return ""

        if answer_format == "tf":
            if "A" in answer:
                return "A"
            if "B" in answer:
                return "B"
            return ""

        if answer_format == "multi":
            letters = sorted(set(c for c in answer if c in valid_letters))
            return "".join(letters) if letters else ""

        return answer
