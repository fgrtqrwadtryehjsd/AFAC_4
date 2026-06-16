"""语义向量索引 — DashScope text-embedding-v3 + FAISS

核心创新：Embedding API不计入比赛Token预算！
- BM25: 关键词匹配（精确但召回差）
- Vector: 语义匹配（模糊但召回好）
- RRF融合: 取长补短

缓存策略：
- 文档块Embedding: cache/vectors/embeddings_768d.npy (已缓存)
- 查询Embedding: cache/vectors/query_cache/ (按查询文本hash缓存)
- 同一查询只调用一次API，后续直接读本地
"""
import os
import json
import time
import hashlib
import numpy as np
from openai import OpenAI
from agent.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, CHUNK_SIZE
from agent.indexer import DocumentIndex

EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 768
BATCH_SIZE = 10  # DashScope embedding批量上限
VECTOR_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "vectors")
QUERY_CACHE_DIR = os.path.join(VECTOR_CACHE_DIR, "query_cache")


def _query_cache_key(text: str) -> str:
    """生成查询缓存key（基于文本hash）"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


class VectorIndexer:
    """语义向量索引器 — 用DashScope Embedding API编码文档块"""

    def __init__(self, doc_index: DocumentIndex):
        self.doc_index = doc_index
        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self.embeddings = None
        self.chunk_ids = None
        self.index = None
        self._query_embed_cache = {}  # 内存缓存 (text → embedding array)
        self._api_call_count = 0      # API调用次数统计
        self._cache_hit_count = 0     # 缓存命中次数
        os.makedirs(QUERY_CACHE_DIR, exist_ok=True)
        self._build_or_load()

    def _build_or_load(self):
        """构建或加载向量索引"""
        os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(VECTOR_CACHE_DIR, f"embeddings_{EMBEDDING_DIM}d.npy")
        meta_file = os.path.join(VECTOR_CACHE_DIR, f"meta_{EMBEDDING_DIM}d.json")

        if os.path.exists(cache_file) and os.path.exists(meta_file):
            print(f"  📦 加载缓存向量索引...")
            self.embeddings = np.load(cache_file)
            with open(meta_file) as f:
                meta = json.load(f)
            self.chunk_ids = meta["chunk_ids"]
            n = len(self.chunk_ids)
            print(f"  向量索引加载完成 ({n} 块, {self.embeddings.shape[1]}d)")
        else:
            print(f"  🔮 构建语义向量索引 (DashScope text-embedding-v3)...")
            self._build_embeddings()
            np.save(cache_file, self.embeddings)
            with open(meta_file, "w") as f:
                json.dump({"chunk_ids": self.chunk_ids}, f)

        self._build_faiss()

        # 加载已有的查询缓存
        self._load_query_cache_index()

    def _load_query_cache_index(self):
        """加载已有的查询缓存索引"""
        cache_index_file = os.path.join(QUERY_CACHE_DIR, "cache_index.json")
        if os.path.exists(cache_index_file):
            with open(cache_index_file) as f:
                self._disk_cache_index = json.load(f)
            print(f"  📦 查询Embedding缓存: {len(self._disk_cache_index)}条")
        else:
            self._disk_cache_index = {}

    def _build_embeddings(self):
        """批量编码所有文档块"""
        chunks = self.doc_index.chunks
        n = len(chunks)
        self.chunk_ids = list(range(n))
        self.embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

        max_chars = 8000

        start = time.time()
        for i in range(0, n, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, n)
            texts = [chunks[j].get("text", "")[:max_chars] for j in range(i, batch_end)]

            try:
                resp = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                    dimensions=EMBEDDING_DIM
                )
                for k, emb in enumerate(resp.data):
                    self.embeddings[i + k] = emb.embedding
            except Exception as e:
                print(f"  ⚠️ Embedding批次 {i}-{batch_end} 失败: {e}")
                pass

            if (batch_end % 500) < BATCH_SIZE or batch_end == n:
                elapsed = time.time() - start
                eta = elapsed / batch_end * (n - batch_end) if batch_end > 0 else 0
                print(f"  进度: {batch_end}/{n} ({batch_end/n*100:.0f}%) "
                      f"{elapsed:.0f}s (ETA {eta:.0f}s)")

        elapsed = time.time() - start
        print(f"  ✅ 向量编码完成: {n}块, {elapsed:.1f}s (不计入比赛Token!)")

    def _build_faiss(self):
        """构建FAISS索引"""
        import faiss
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.embeddings / norms
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(normalized.astype(np.float32))
        self.normalized_embeddings = normalized

    def _embed_query(self, text: str) -> np.ndarray:
        """编码查询文本（带本地缓存，避免重复API调用）

        缓存层级：
        1. 内存缓存 _query_embed_cache（最快）
        2. 磁盘缓存 cache/vectors/query_cache/（持久化）
        3. API调用（最慢，首次查询时）
        """
        cache_key = _query_cache_key(text)

        # Level 1: 内存缓存
        if cache_key in self._query_embed_cache:
            self._cache_hit_count += 1
            return self._query_embed_cache[cache_key]

        # Level 2: 磁盘缓存
        if cache_key in self._disk_cache_index:
            npy_path = os.path.join(QUERY_CACHE_DIR, f"{cache_key}.npy")
            if os.path.exists(npy_path):
                emb = np.load(npy_path)
                self._query_embed_cache[cache_key] = emb
                self._cache_hit_count += 1
                return emb

        # Level 3: API调用
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[text[:8000]],
                    dimensions=EMBEDDING_DIM,
                    timeout=30.0
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f" [VEC_ERR:{e}]")
                    return np.zeros(EMBEDDING_DIM, dtype=np.float32)
                time.sleep(2)

        self._api_call_count += 1
        emb = np.array(resp.data[0].embedding, dtype=np.float32)

        # 存入缓存
        self._query_embed_cache[cache_key] = emb
        np.save(os.path.join(QUERY_CACHE_DIR, f"{cache_key}.npy"), emb)
        self._disk_cache_index[cache_key] = text[:200]  # 记录原文预览
        # 每隔20次写入索引
        if self._api_call_count % 20 == 0:
            self._save_cache_index()

        return emb

    def _save_cache_index(self):
        """保存查询缓存索引到磁盘"""
        with open(os.path.join(QUERY_CACHE_DIR, "cache_index.json"), "w") as f:
            json.dump(self._disk_cache_index, f, ensure_ascii=False, indent=2)

    def search_vector(self, query: str, top_k: int = 20, doc_ids: list = None) -> list:
        """语义向量搜索（带本地缓存）

        Returns: list of (chunk_idx, score) sorted by score desc
        """
        q_emb = self._embed_query(query)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores, indices = self.index.search(q_emb.reshape(1, -1), min(top_k * 5, len(self.chunk_ids)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.doc_index.chunks[idx]
            if doc_ids and chunk.get("doc_id") not in doc_ids:
                continue
            results.append((int(idx), float(score)))

        return results[:top_k]

    def search_vector_multi_query(self, queries: list, top_k: int = 20, doc_ids: list = None) -> list:
        """多查询向量搜索 — 合并为单文本后一次Embedding"""
        combined_query = " ".join(queries)
        combined_query = combined_query[:8000]

        q_emb = self._embed_query(combined_query)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        k_search = min(top_k * 3, len(self.chunk_ids))
        scores, indices = self.index.search(q_emb.reshape(1, -1), k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.doc_index.chunks[idx]
            if doc_ids and chunk.get("doc_id") not in doc_ids:
                continue
            results.append((int(idx), float(score)))

        return results[:top_k]

    def rrf_fuse(self, bm25_results: list, vector_results: list,
                 k: int = 60, top_n: int = 30) -> list:
        """RRF (Reciprocal Rank Fusion) 融合BM25和向量搜索结果"""
        rrf_scores = {}

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return sorted_results[:top_n]

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return {
            "api_calls": self._api_call_count,
            "cache_hits": self._cache_hit_count,
            "disk_cache_size": len(self._disk_cache_index),
            "mem_cache_size": len(self._query_embed_cache),
        }

    def finalize(self):
        """结束时保存所有缓存"""
        self._save_cache_index()
