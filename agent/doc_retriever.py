"""B 榜文档级检索器 — 无 doc_ids 时自动检索候选文档.

## 背景
A 榜题目给 doc_ids, B 榜不给 (需先检索候选文档再问答).
B 榜 7/22 才开放, 现在用 A 榜题目做"留出测试"验证检索召回率.

## 策略 (融合论文 + 工业实践)
1. **domain 过滤** (Contextual Retrieval 思路): 题目有 domain 字段, 先把候选池缩到该域.
   - financial_contracts(14) / financial_reports(10) / insurance(16) / regulatory(513) / research(20)
   - 把 regulatory 513 文档的检索变可管理.
2. **多路 BM25 chunk 检索** (Self-Ask 多查询): 题干 + 每个选项分别检索, 提高召回.
3. **doc 级 RRF 聚合** (LongRAG doc-level grouping): chunk 命中按 RRF 分数聚合到 doc_id,
   而非直接返回散点 chunk. 取 top-k doc_ids 喂给问答 (与 A 榜 doc_ids 用法一致).
4. **可选向量融合**: VectorIndexer RRF 融合 BM25 (embedding 不计比赛 token, 但花 DashScope 费,
   多数查询有缓存). 默认关闭, 留作增强.

## 零成本保证
纯 BM25 路径: 本地 jieba 分词 + BM25Okapi, 零 API 调用, 零 token.
向量路径: embedding 不占比赛 token 预算, 但调 DashScope embedding API (极廉价, 多数缓存命中).
"""
import os
import re
from collections import defaultdict
from agent.config import PROCESSED_DIR

# 文档头部实体名匹配优先域 (题干点名产品/公司/法规全名, 头部精确匹配 > BM25 词频)
ENTITY_MATCH_DOMAINS = {"insurance", "financial_reports", "regulatory"}

# 金融常见实体后缀: 提取题干中的产品/公司/法规名
_ENTITY_SUFFIXES = (
    "保险", "养老保险", "年金保险", "医疗保险", "寿险", "疾病保险",
    "意外伤害保险", "财产保险", "责任保险", "集团", "股份", "时代",
    "办法", "管理规定", "管理条例", "法", "规定",
)
# 干扰词: 这些是题干通用词, 不是实体名
_STOP_WORDS = {"公司", "保险", "金融", "机构", "下列", "以下", "关于", "结合",
               "根据", "哪些", "说法", "描述", "陈述", "结论", "正确", "准确",
               "产品", "两份", "报告", "年度报告", "募集说明书", "条款"}


class DocRetriever:
    """文档级检索器: question (无 doc_ids) → top-k 候选 doc_ids"""

    def __init__(self, doc_index, vector_indexer=None):
        self.doc_index = doc_index
        self.vector = vector_indexer
        # doc_id -> domain 映射 (从 processed_data 目录结构建, 零成本)
        self.doc_domain = {}
        # domain -> [doc_id] 反向索引
        self.domain_docs = defaultdict(list)
        self._build_doc_domain_map()
        # doc_id -> 文档头部 500 字 (实体名匹配用, 缓存)
        self._doc_head_cache = {}

    def _doc_head(self, doc_id, n=500):
        """取文档头部 n 字符 (产品/公司/法规名通常在头部)."""
        if doc_id not in self._doc_head_cache:
            txt = self.doc_index.get_doc_full_text(doc_id) or ""
            self._doc_head_cache[doc_id] = txt[:n]
        return self._doc_head_cache[doc_id]

    @staticmethod
    def _extract_entities(text):
        """从题干/选项提取实体名 (产品/公司/法规全名).

        策略: 提取书名号《》内内容 + 含实体后缀的连续中文片段.
        """
        entities = set()
        # 1. 书名号内内容 (法规名常见, 如《...办法》)
        for m in re.findall(r"《([^》]{2,40})》", text):
            entities.add(m.strip())
        # 2. 含后缀的连续中文/字母数字片段 (产品/公司名)
        #    匹配 "平安智盈金生""宁德时代""比亚迪" 这类
        #    用后缀锚定, 避免抓到通用词
        for suf in _ENTITY_SUFFIXES:
            pattern = r"([一-龥A-Za-z0-9]{2,12}" + re.escape(suf) + r")"
            for m in re.findall(pattern, text):
                m = m.strip()
                # 过滤: 纯后缀词或停用词
                if m in _STOP_WORDS or m == suf:
                    continue
                # 过滤: 去掉后缀后剩余 < 2 字的 (如"公司"本身)
                core = m[: -len(suf)] if m.endswith(suf) else m
                if len(core) < 2:
                    continue
                entities.add(m)
        return entities

    def _build_doc_domain_map(self):
        """从 processed_data/{domain}/{doc_id}.json 目录结构建映射."""
        for domain in os.listdir(PROCESSED_DIR):
            ddir = os.path.join(PROCESSED_DIR, domain)
            if not os.path.isdir(domain) and not os.path.isdir(ddir):
                continue
            if not os.path.isdir(ddir):
                continue
            for root, dirs, files in os.walk(ddir):
                for f in files:
                    if f.endswith(".json") and f != "structured_index.json":
                        did = os.path.splitext(f)[0]
                        self.doc_domain[did] = domain
                        self.domain_docs[domain].append(did)

    def _candidate_pool(self, domain):
        """返回候选文档 id 集合 (按 domain 过滤, 无 domain 则全库)."""
        if domain and domain in self.domain_docs:
            return self.domain_docs[domain]
        return list(self.doc_domain.keys())

    @staticmethod
    def _rrf_score(rank, k=60):
        """RRF 单路得分: 1/(k+rank+1)."""
        return 1.0 / (k + rank + 1)

    def retrieve_docs(self, question, top_k=3, use_vector=False):
        """检索候选文档.

        Args:
            question: 题目 dict (用 question/options/domain 字段)
            top_k: 返回文档数 (A 榜多数题 doc_ids ≤ 4, 默认 3)
            use_vector: 是否融合向量检索 (默认 False, 纯 BM25 零成本)

        Returns:
            list[str] doc_ids, 按相关度降序
        """
        domain = question.get("domain")
        q_text = question.get("question", "")
        options = question.get("options", {})
        pool = self._candidate_pool(domain)
        pool_set = set(pool)

        # 多路查询: 题干 + 每个选项 (Self-Ask 多查询提高召回)
        queries = [q_text]
        for k in sorted(options.keys()):
            opt = options.get(k, "")
            if opt and opt.strip():
                queries.append(f"{q_text} {opt}")

        doc_scores = defaultdict(float)

        # 0. 实体名精确匹配前置路 (Contextual Retrieval: 头部即实体上下文)
        #    题干点名的产品/公司/法规名 → 文档头部子串匹配, 命中给高加分.
        #    直接解决 BM25 词频虚高分 (如 doc6 干扰 insurance 检索).
        if domain in ENTITY_MATCH_DOMAINS:
            full_q = q_text + " " + " ".join(str(v) for v in options.values())
            entities = self._extract_entities(full_q)
            for did in pool:
                head = self._doc_head(did)
                if not head:
                    continue
                hits = 0
                for ent in entities:
                    if ent and ent in head:
                        hits += 1
                if hits > 0:
                    # 每命中一个实体 +1.0 (远大于单路 RRF ~0.016)
                    # 命中越多说明文档越相关 (多产品对比题点多个名)
                    doc_scores[did] += 2.0 * hits

        # BM25 多路检索, doc 级聚合用 max(chunk_score) 归一化 + RRF 辅助
        # 关键修正: RRF 只看 rank 丢掉"独有词高分"信号 (如保险产品名独占一文档,
        # 单 chunk BM25 分 170 却因只命中 1 个 chunk 被 RRF 压到第 6).
        # 解法 (LongRAG doc-level): doc 主分 = 各查询命中该 doc 的 max chunk_score,
        # 归一化到 [0,1] 后加权; RRF 作为辅助平滑分.
        doc_max_score = defaultdict(float)   # did -> max(各查询该 doc 的 chunk score)
        for query in queries:
            if not query.strip():
                continue
            chunks = self.doc_index.search_bm25(query, top_k=30, doc_ids=pool)
            if not chunks:
                continue
            # 该查询内 chunk score 归一化 (除以本查询 max, 避免跨查询分数不可比)
            q_max = max(c.get("score", 0) for c in chunks) or 1.0
            for rank, chunk in enumerate(chunks):
                did = chunk.get("doc_id")
                if did in pool_set:
                    norm = chunk.get("score", 0) / q_max   # [0,1]
                    # 主信号: 该 doc 在本查询的归一化最高分
                    if norm > doc_max_score[did]:
                        doc_max_score[did] = norm
                    # 辅助信号: RRF (平滑, 多 chunk 命中略加分)
                    doc_scores[did] += self._rrf_score(rank) * 0.3

        # 可选向量融合 (embedding 不占比赛 token, 但花 DashScope 费; 默认关闭)
        # 向量也走 max 归一化, 与 BM25 主信号同量级叠加
        if use_vector and self.vector is not None:
            vec_max = defaultdict(float)
            for query in queries:
                if not query.strip():
                    continue
                v_results = self.vector.search_vector(query, top_k=30, doc_ids=pool)
                if not v_results:
                    continue
                q_max = max(s for _, s in v_results) or 1.0
                for rank, (chunk_idx, score) in enumerate(v_results):
                    chunk = self.doc_index.chunks[chunk_idx]
                    did = chunk.get("doc_id")
                    if did in pool_set:
                        norm = score / q_max
                        if norm > vec_max[did]:
                            vec_max[did] = norm
                        doc_scores[did] += self._rrf_score(rank) * 0.3
            for did in pool_set:
                doc_max_score[did] += vec_max.get(did, 0.0)

        # 合并: 主信号 (max 归一化分) + 实体匹配分 + RRF 辅助
        # 实体匹配分 + RRF 辅助分已在 doc_scores 里
        final_scores = {}
        for did in pool_set:
            final_scores[did] = doc_max_score[did] + doc_scores.get(did, 0.0)
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])
        return [did for did, _ in ranked[:top_k]]

    def resolve_doc_ids(self, question, top_k=3, use_vector=False):
        """兼容入口: 若题目已有 doc_ids 则直接返回, 否则检索.

        pipeline 调用: doc_ids = retriever.resolve_doc_ids(question)
        A 榜 (有 doc_ids) → 原样返回; B 榜 (无 doc_ids) → 自动检索.
        """
        existing = question.get("doc_ids") or []
        if existing:
            return list(existing)
        return self.retrieve_docs(question, top_k=top_k, use_vector=use_vector)
