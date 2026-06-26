"""V13.1: V13 + doc 级配额 + 同义词扩展.

V13 的两个被诊断出的瓶颈:
1. 多 doc 题 RRF 全局排序导致短/异形 doc 完全消失 (fin_a_012 比亚迪 0K, ins_a_007 国寿 1K)
2. 同义词错配:题干"贷款" vs 文档"借款" (ins_a_007)

修复:
1. 检索后按 doc_id 分桶,每 doc 保证最低字符配额,然后填满剩余配额
2. query 端用 synonym_expander 扩展 BM25/keyword 查询词
3. **不改 prompt 结构、不引入额外 LLM 调用、不改阈值** — 严格单变量
"""

from agent.reasoner_v13 import (
    ReasoningAgentV13,
    DOMAIN_SYSTEM_PROMPTS,
    COT_PROMPT,
    SELF_CRITIQUE_PROMPT,
    CLAUSE_PATTERN,
    extract_query_keywords,
    keyword_match_score,
    compress_whitespace,
)
from agent.synonym_expander import expand_query_text, expand_query_terms
from agent.postprocessor import extract_answer_from_response
from collections import defaultdict


class ReasoningAgentV131(ReasoningAgentV13):
    """V13.1: doc 级配额 + 同义词扩展"""

    # 每 doc 至少占总证据的 (1/n_docs * MIN_DOC_SHARE_RATIO),例如 2 doc 题至少各 30%
    MIN_DOC_SHARE_RATIO = 0.6

    def _retrieve_merged_evidence(self, q_text: str, options: dict,
                                   doc_ids: list, card_hints: dict,
                                   answer_format: str) -> str:
        """V13.1: 三路检索全部用同义词扩展查询; 融合后按 doc 分桶填证据,保证 doc 级配额"""
        max_evidence = self.EVIDENCE_LIMITS.get(answer_format, 20000)

        # ★ 同义词扩展 query
        q_expanded = expand_query_text(q_text)

        # 1. BM25 主查询 (扩展词)
        bm25_results = self.doc_index.search_bm25(q_expanded, top_k=20, doc_ids=doc_ids)

        # 2. 语义向量检索 (用原 q_text — vector 模型自己处理语义,不需要词扩展)
        vector_chunks = []
        if self.vector:
            q_results = self.vector.search_vector(q_text, top_k=15, doc_ids=doc_ids)
            opt_queries = [f"{q_text} {options[k]}" for k in sorted(options.keys())]
            opt_results = self.vector.search_vector_multi_query(
                opt_queries, top_k=15, doc_ids=doc_ids)
            fused_vector = self.vector.rrf_fuse(q_results, opt_results, top_n=25)
            for idx, score in fused_vector:
                if idx < len(self.doc_index.chunks):
                    chunk = self.doc_index.chunks[idx]
                    if chunk.get("doc_id") in doc_ids:
                        vector_chunks.append(chunk)

        # 3. Card 引导关键词检索 (★ 用扩展词)
        base_keywords = extract_query_keywords(q_text, options)
        # ★ 把题干+选项的同义词扩展词加进来
        all_text = q_text + " " + " ".join(options.values())
        synonym_keywords = expand_query_terms(all_text)
        search_keywords = list(set(
            base_keywords +
            synonym_keywords +
            card_hints.get("entity_terms", []) +
            card_hints.get("key_term_hits", [])
        ))
        all_doc_chunks = self.doc_index.get_chunks_by_doc_ids(doc_ids)
        kw_scored = []
        for chunk in all_doc_chunks:
            text = chunk.get("text", "")
            score = keyword_match_score(search_keywords, text)
            if score > 0:
                kw_scored.append((score, chunk))
        kw_scored.sort(key=lambda x: -x[0])
        kw_results = [chunk for _, chunk in kw_scored[:60]]

        # 4. Card 条款精准定位
        clause_evidence = []
        for doc_id in doc_ids:
            for clause_ref in card_hints.get("clause_refs", []):
                clause_text = self.memory.locate_clause_text(doc_id, clause_ref, max_len=3000)
                if clause_text:
                    clause_evidence.append({
                        "doc_id": doc_id, "text": clause_text,
                        "chunk_type": "clause_precise", "clause_ref": clause_ref,
                    })

        # 5. 数值型精准定位
        metric_evidence = []
        for doc_id in doc_ids:
            for metric_val in card_hints.get("metric_contexts", []):
                metric_ctx = self.memory.locate_metric_context(doc_id, metric_val)
                if metric_ctx:
                    metric_evidence.append({
                        "doc_id": doc_id, "text": metric_ctx,
                        "chunk_type": "metric_precise",
                    })

        # 6. 三路 RRF 融合
        bm25_kw = self._rrf_fuse_chunks(
            bm25_results, kw_results,
            bm25_weight=1.0, kw_weight=1.5, top_n=40)
        final_chunks = self._rrf_fuse_chunks_3way(
            clause_evidence + metric_evidence, bm25_kw, vector_chunks, top_n=60)

        # 7. ★ doc 级配额组装
        evidence_text = self._assemble_with_doc_quota(
            final_chunks, doc_ids, max_evidence
        )

        if len(evidence_text) > max_evidence:
            evidence_text = evidence_text[:max_evidence] + "\n[...证据已截断...]"

        return evidence_text

    def _assemble_with_doc_quota(self, chunks, doc_ids, max_evidence):
        """按 doc 分桶填充,保证每 doc 至少占用 (max/n_docs * MIN_DOC_SHARE_RATIO) 字符,
        且最多不超过 (max/n_docs * MAX_DOC_SHARE_RATIO) — 防止单 doc 撑爆.

        Phase 1: 每 doc 按融合后排名挑 chunks,直到达到该 doc 的最小配额
        Phase 2: 按全局排名填,但每 doc 单独有上限
        """
        n_docs = max(1, len(doc_ids))
        min_quota = int(max_evidence / n_docs * self.MIN_DOC_SHARE_RATIO)
        max_quota = int(max_evidence / n_docs * 1.6)  # 上限:平均的 1.6 倍

        chunks_by_doc = defaultdict(list)
        for c in chunks:
            d = c.get("doc_id", "")
            if d in doc_ids:
                chunks_by_doc[d].append(c)

        seen_keys = set()
        per_doc_used = defaultdict(int)
        segments = []

        # Phase 1: 每 doc 最低配额
        for d in doc_ids:
            for c in chunks_by_doc[d]:
                text = c.get("text", "")
                ref = c.get("clause_ref", "")
                key = text[:100]
                if key in seen_keys:
                    continue
                if per_doc_used[d] >= min_quota:
                    break
                # ★ 若该 chunk 加进来会超过该 doc 的 max_quota,截断
                remain = max_quota - per_doc_used[d]
                if len(text) > remain and remain > 0:
                    text = text[:remain] + "...[截断]"
                seen_keys.add(key)
                segments.append((d, ref, text))
                per_doc_used[d] += len(text)

        # Phase 2: 全局填剩余空间,但 per_doc 不超过 max_quota
        used_total = sum(per_doc_used.values())
        for c in chunks:
            if used_total >= max_evidence:
                break
            text = c.get("text", "")
            key = text[:100]
            if key in seen_keys:
                continue
            d = c.get("doc_id", "")
            if d not in doc_ids:
                continue
            if per_doc_used[d] >= max_quota:
                continue
            # ★ 截断超额部分
            remain = max_quota - per_doc_used[d]
            if len(text) > remain and remain > 0:
                text = text[:remain] + "...[截断]"
            seen_keys.add(key)
            ref = c.get("clause_ref", "")
            segments.append((d, ref, text))
            used_total += len(text)
            per_doc_used[d] += len(text)

        # 按 doc_id 顺序输出,提升"按 doc 推理"
        segments.sort(key=lambda s: doc_ids.index(s[0]) if s[0] in doc_ids else 999)

        evidence_text = ""
        for d, ref, text in segments:
            label = f"文档 {d}" + (f" [{ref}]" if ref else "")
            evidence_text += f"\n--- 来自{label} ---\n{text}\n"

        return evidence_text
