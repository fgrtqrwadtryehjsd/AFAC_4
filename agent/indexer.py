"""文档索引与检索模块 V2

使用结构化索引 + BM25 + Qwen 语义 Rerank
"""
import os
import json
import jieba
from rank_bm25 import BM25Okapi
from agent.config import PROCESSED_DIR, TOP_K_CHUNKS
from agent.qwen_client import QwenClient


class DocumentIndex:
    """文档索引管理器 V2"""

    def __init__(self, qwen_client: QwenClient = None):
        self.documents = {}      # doc_id -> full_text
        self.doc_meta = {}       # doc_id -> metadata
        self.chunks = []         # [{doc_id, chunk_id, text, type}]
        self.bm25 = None
        self.bm25_corpus = []
        self.qwen = qwen_client
        self._loaded = False

    def load(self):
        """加载预处理文档和结构化索引"""
        if self._loaded:
            return

        # 1. 加载全文
        self._load_full_texts()

        # 2. 加载结构化分块索引
        index_path = os.path.join(PROCESSED_DIR, "structured_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            self.doc_meta = index_data.get("doc_meta", {})
            self.chunks = index_data.get("chunks", [])
            print(f"  加载结构化索引: {len(self.chunks)} 个块")
        else:
            # 降级：固定窗口分块
            self._build_fixed_chunks()

        # 3. 构建 BM25
        self._build_bm25()
        self._loaded = True

    def _load_full_texts(self):
        """加载全文数据"""
        for domain in os.listdir(PROCESSED_DIR):
            domain_dir = os.path.join(PROCESSED_DIR, domain)
            if not os.path.isdir(domain_dir):
                continue
            for root, dirs, files in os.walk(domain_dir):
                for f in files:
                    if not f.endswith('.json') or f == "structured_index.json":
                        continue
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, "r", encoding="utf-8") as fh:
                            doc_data = json.load(fh)
                        doc_id = os.path.splitext(f)[0]
                        self.documents[doc_id] = doc_data.get("full_text", "")
                    except:
                        pass
        print(f"  加载了 {len(self.documents)} 个文档全文")

    def _build_fixed_chunks(self):
        """降级：固定窗口分块"""
        self.chunks = []
        chunk_id = 0
        chunk_size = 2000
        overlap = 200
        for doc_id, text in self.documents.items():
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if chunk_text.strip():
                    self.chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "type": "fixed",
                    })
                    chunk_id += 1

    def _build_bm25(self):
        """构建 BM25 索引"""
        self.bm25_corpus = []
        for chunk in self.chunks:
            tokens = list(jieba.cut(chunk["text"]))
            self.bm25_corpus.append(tokens)
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
            print(f"  BM25 索引构建完成 ({len(self.bm25_corpus)} 块)")

    def search_bm25(self, query: str, doc_ids: list = None, top_k: int = None) -> list:
        """BM25 检索"""
        if not self.bm25:
            return []
        top_k = top_k or TOP_K_CHUNKS * 2  # 多召回一些给 rerank
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        
        candidates = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if doc_ids and chunk["doc_id"] not in doc_ids:
                continue
            candidates.append((score, idx, chunk))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[2] for c in candidates[:top_k]]

    def search_with_rerank(self, query: str, doc_ids: list = None, top_k: int = None) -> list:
        """
        BM25 粗召回 + Qwen 精排 (Rerank)
        
        流程：
        1. BM25 召回 top_k*2 候选
        2. 用 Qwen 判断每个候选与问题的相关性
        3. 只保留相关证据，按相关度排序
        """
        top_k = top_k or TOP_K_CHUNKS
        
        # Step 1: BM25 粗召回
        candidates = self.search_bm25(query, doc_ids, top_k=top_k * 3)
        
        if not candidates:
            return []
        
        # Step 2: Qwen Rerank（如果 token 预算充足）
        if self.qwen and len(candidates) > top_k:
            candidates = self._qwen_rerank(query, candidates, top_k)
        
        return candidates[:top_k]

    def _qwen_rerank(self, query: str, candidates: list, top_k: int) -> list:
        """用 Qwen 对候选块做相关性判断"""
        # 为节省 token，只取每块前 300 字做判断
        excerpts = []
        for i, chunk in enumerate(candidates):
            excerpt = chunk["text"][:300]
            excerpts.append(f"[{i}] 来源:{chunk['doc_id']}\n{excerpt}")
        
        prompt = f"""判断以下文档片段与问题的相关性，返回最相关的 {top_k} 个片段编号。

问题：{query}

文档片段：
{chr(10).join(excerpts[:15])}

请输出最相关的片段编号，用逗号分隔，例如：0,3,7
只输出编号，不要解释。"""

        try:
            result = self.qwen.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=50
            )
            # 解析输出
            content = result["content"].strip()
            indices = [int(x.strip()) for x in content.split(",") if x.strip().isdigit()]
            indices = [i for i in indices if 0 <= i < len(candidates)]
            
            if indices:
                reranked = [candidates[i] for i in indices]
                # 补充未被选中的（防止遗漏）
                remaining = [c for c in candidates if c not in reranked]
                return reranked + remaining
        except:
            pass
        
        return candidates

    def search_by_options(self, question: str, options: dict, doc_ids: list = None, top_k: int = None) -> list:
        """多查询检索：题干 + 各选项分别检索，合并去重"""
        top_k = top_k or TOP_K_CHUNKS
        seen_ids = set()
        all_chunks = []
        
        # 题干检索
        for chunk in self.search_with_rerank(question, doc_ids, top_k=top_k):
            cid = chunk.get("chunk_id", id(chunk))
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)
        
        # 各选项检索（补充证据）
        for opt_key, opt_text in options.items():
            query = f"{question} {opt_text}"
            for chunk in self.search_with_rerank(query, doc_ids, top_k=2):
                cid = chunk.get("chunk_id", id(chunk))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(chunk)
        
        return all_chunks

    def get_doc_full_text(self, doc_id: str) -> str:
        return self.documents.get(doc_id, "")

    def get_doc_chunks(self, doc_id: str) -> list:
        """获取某文档的所有块"""
        return [c for c in self.chunks if c["doc_id"] == doc_id]
