"""语义向量索引 — DashScope text-embedding-v3 + FAISS

核心创新：Embedding API不计入比赛Token预算！
- BM25: 关键词匹配（精确但召回差）
- Vector: 语义匹配（模糊但召回好）
- RRF融合: 取长补短
"""
import os
import json
import time
import numpy as np
from openai import OpenAI
from agent.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, CHUNK_SIZE
from agent.indexer import DocumentIndex

EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 768  # 降低维度减少存储和提高速度
BATCH_SIZE = 10  # DashScope embedding批量上限
VECTOR_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "vectors")


class VectorIndexer:
    """语义向量索引器 — 用DashScope Embedding API编码文档块"""

    def __init__(self, doc_index: DocumentIndex):
        self.doc_index = doc_index
        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self.embeddings = None  # (n_chunks, dim)
        self.chunk_ids = None   # list of chunk indices
        self.index = None       # FAISS index
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

    def _build_embeddings(self):
        """批量编码所有文档块"""
        chunks = self.doc_index.chunks
        n = len(chunks)
        self.chunk_ids = list(range(n))
        self.embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

        # 截断过长文本 (embedding模型有token限制)
        max_chars = 8000  # ~4000 tokens

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
                # 用零向量占位
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

        # L2归一化 → 用内积等价余弦相似度
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # 避免除零
        normalized = self.embeddings / norms

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # 内积索引
        self.index.add(normalized.astype(np.float32))
        self.normalized_embeddings = normalized

    def search_vector(self, query: str, top_k: int = 20, doc_ids: list = None) -> list:
        """语义向量搜索（带超时和重试）

        Returns: list of (chunk_idx, score) sorted by score desc
        """
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[query[:8000]],
                    dimensions=EMBEDDING_DIM,
                    timeout=30.0
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f" [VEC_ERR:{e}]")
                    return []
                import time; time.sleep(2)
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        # 搜索
        scores, indices = self.index.search(q_emb.reshape(1, -1), min(top_k * 5, len(self.chunk_ids)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.doc_index.chunks[idx]
            # 如果指定了doc_ids，过滤
            if doc_ids and chunk.get("doc_id") not in doc_ids:
                continue
            results.append((int(idx), float(score)))

        return results[:top_k]

    def search_vector_multi_query(self, queries: list, top_k: int = 20, doc_ids: list = None) -> list:
        """多查询向量搜索 — 一次Embedding API调用（合并为单文本）
        """
        combined_query = " ".join(queries)
        combined_query = combined_query[:8000]

        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[combined_query],
                    dimensions=EMBEDDING_DIM,
                    timeout=30.0
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f" [VEC_MULTI_ERR:{e}]")
                    return []
                import time; time.sleep(2)
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        # 搜索
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
        """RRF (Reciprocal Rank Fusion) 融合BM25和向量搜索结果

        bm25_results: [(chunk_idx, score), ...]
        vector_results: [(chunk_idx, score), ...]
        Returns: [(chunk_idx, rrf_score), ...] sorted by rrf_score desc
        """
        rrf_scores = {}

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return sorted_results[:top_n]
