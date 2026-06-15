"""RAG 检索模块 + 上下文窗口优化

核心技术：
1. RAG：BM25 关键词检索 + 结构化索引辅助
2. 上下文窗口优化：根据 token 预算动态裁剪证据，优先保留高相关度内容
3. 多路召回：问题检索 + 选项检索 + 条款精准定位
"""
import os
import re
import json
import math
from collections import Counter
from agent.config import PROCESSED_DIR
from agent.clause_locator import extract_clause_refs, locate_clauses_in_doc


class DocumentIndex:
    """文档索引（BM25 + 结构化）"""

    def __init__(self):
        self.full_texts = {}   # doc_id -> full_text
        self.chunks = []       # 所有结构化块
        self.bm25 = None
        self.doc_lengths = {}  # doc_id -> char_count

    def load(self):
        """加载文档和索引"""
        # 加载全文
        for domain in os.listdir(PROCESSED_DIR):
            domain_dir = os.path.join(PROCESSED_DIR, domain)
            if not os.path.isdir(domain_dir) or domain in ("compressed_memory", "structured_extracts"):
                continue
            for root, dirs, files in os.walk(domain_dir):
                for f in files:
                    if not f.endswith('.json') or f == "structured_index.json":
                        continue
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        doc_id = os.path.splitext(f)[0]
                        text = data.get("full_text", "")
                        if text:
                            self.full_texts[doc_id] = text
                            self.doc_lengths[doc_id] = len(text)
                    except:
                        pass

        # 加载结构化索引
        index_path = os.path.join(PROCESSED_DIR, "structured_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            # 兼容两种格式：list 或 {"chunks": [...]}
            if isinstance(index_data, dict):
                self.chunks = index_data.get("chunks", [])
            elif isinstance(index_data, list):
                self.chunks = index_data

        # 构建 BM25
        self._build_bm25()
        print(f"  加载了 {len(self.full_texts)} 个文档，{len(self.chunks)} 个块")

    def _build_bm25(self):
        """构建 BM25 索引"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba
        except ImportError:
            print("  BM25 不可用，跳过检索")
            self.bm25 = None
            return

        tokenized = []
        for chunk in self.chunks:
            text = chunk if isinstance(chunk, str) else chunk.get("text", "")
            tokens = list(jieba.cut(text))
            tokenized.append(tokens)

        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 索引构建完成 ({len(self.chunks)} 块)")

    def get_doc_full_text(self, doc_id: str) -> str:
        return self.full_texts.get(doc_id, "")

    def get_chunks_by_doc_ids(self, doc_ids: list) -> list:
        """获取指定文档的所有 chunks"""
        return [c for c in self.chunks if isinstance(c, dict) and c.get("doc_id") in doc_ids]

    def search(self, query: str, top_k: int = 10, doc_ids: list = None) -> list:
        """BM25 检索（兼容旧接口）"""
        return self.search_bm25(query, top_k=top_k, doc_ids=doc_ids)

    def search_bm25(self, query: str, top_k: int = 10, doc_ids: list = None) -> list:
        """BM25 检索，可限定文档范围"""
        if not self.bm25:
            return []

        import jieba
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)

        # 按分数排序
        scored_chunks = list(zip(scores, self.chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen = set()
        for score, chunk in scored_chunks:
            if score <= 0:
                break
            if doc_ids and chunk.get("doc_id") not in doc_ids:
                continue
            # 去重
            text_key = chunk.get("text", "")[:100]
            if text_key in seen:
                continue
            seen.add(text_key)
            results.append({**chunk, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

    def search_with_clause_location(self, question: str, options: dict, doc_ids: list, top_k: int = 8) -> list:
        """多路召回：BM25 + 选项检索 + 条款精准定位"""
        all_chunks = []
        seen = set()

        def add_chunks(chunks):
            for c in chunks:
                key = c.get("text", "")[:100]
                if key not in seen:
                    seen.add(key)
                    all_chunks.append(c)

        # 1. BM25 主查询
        bm25_results = self.search_bm25(question, top_k=top_k, doc_ids=doc_ids if doc_ids else None)
        add_chunks(bm25_results)

        # 2. 选项检索
        for opt_key, opt_text in options.items():
            opt_results = self.search_bm25(f"{question} {opt_text}", top_k=2, doc_ids=doc_ids if doc_ids else None)
            add_chunks(opt_results)

        # 3. 条款精准定位
        clause_refs = extract_clause_refs(question + " " + " ".join(options.values()))
        if clause_refs and doc_ids:
            for doc_id in doc_ids:
                full_text = self.get_doc_full_text(doc_id)
                if full_text:
                    clause_texts = locate_clauses_in_doc(clause_refs, full_text)
                    for ref, text in clause_texts.items():
                        add_chunks([{"doc_id": doc_id, "text": text, "chunk_type": "clause", "clause_ref": ref}])

        return all_chunks[:top_k + 5]


class ContextWindowOptimizer:
    """上下文窗口优化：根据 token 预算动态裁剪证据"""

    def __init__(self, max_context_chars: int = 12000):
        self.max_chars = max_context_chars

    def optimize(self, evidence_chunks: list, compressed_memory: str = "", max_chars: int = None) -> str:
        """优化上下文：优先级 压缩记忆 > 条款精准 > BM25高分
        
        在 token 预算内最大化有效信息量
        """
        max_chars = max_chars or self.max_chars
        parts = []
        total = 0

        # 1. 压缩记忆（高优先级，但限制长度）
        if compressed_memory:
            mem_len = min(len(compressed_memory), max_chars // 2)
            mem_text = compressed_memory[:mem_len]
            parts.append(f"### 文档摘要\n{mem_text}")
            total += mem_len

        # 2. 条款精准定位（高优先级）
        clause_chunks = [c for c in evidence_chunks if c.get("chunk_type") == "clause"]
        for c in clause_chunks:
            if total >= max_chars:
                break
            text = c["text"][:1500]
            parts.append(f"### 条款定位：{c.get('clause_ref', '?')}\n{text}")
            total += len(text)

        # 3. BM25 高分块
        bm25_chunks = sorted(
            [c for c in evidence_chunks if c.get("chunk_type") != "clause"],
            key=lambda c: c.get("score", 0), reverse=True
        )
        for c in bm25_chunks:
            if total >= max_chars:
                break
            remaining = max_chars - total
            text = c["text"][:min(800, remaining)]
            parts.append(f"### 证据（来源：{c.get('doc_id', '?')}）\n{text}")
            total += len(text)

        return "\n\n".join(parts) if parts else ""
