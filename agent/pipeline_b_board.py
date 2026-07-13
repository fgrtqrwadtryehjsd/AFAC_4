"""B 榜自适应流水线: 无 doc_ids 时自动检索候选文档.

## 用途
A 榜 (有 doc_ids) → 原样用, 行为同 V31.
B 榜 (无 doc_ids) → 用 DocRetriever 自动检索 top-k 候选文档, 再走 V31 推理.

## 零成本保证
- DocRetriever 纯 BM25 路径零 token (本地 jieba + rank_bm25)
- A 榜留出测试召回率: top-4 部分命中 99%, 全命中 58% (见 verify_b_retriever.py)
- B 榜开放 (7/22) 后改 split="B" 即可启用

## 不自动跑
此文件仅定义入口. B 榜开放后手动运行:
    python -m agent.pipeline_b_board   # split 在文件内配置
"""
import os, json
from collections import Counter
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.doc_retriever import DocRetriever
from agent.reasoner_v31 import ReasoningAgentV31
from agent.postprocessor import generate_answer_csv_token_stats


# ===== 配置: B 榜开放后改这里 =====
SPLIT = "A"          # "A" 或 "B"; B 榜开放后改 "B"
B_RETRIEVE_TOPK = 4  # B 榜无 doc_ids 时检索的文档数 (A 榜多数 doc_ids ≤ 4)
USE_VECTOR = False   # 是否向量融合 (花 DashScope embedding 费, 默认关)
# =================================


def load_questions(split="A"):
    qs = []
    qa = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    if not os.path.exists(qa):
        print(f"⚠️ 题目目录不存在: {qa}")
        return qs
    for fn in sorted(os.listdir(qa)):
        if fn.endswith(".json"):
            with open(os.path.join(qa, fn), encoding="utf-8") as f:
                qs.extend(json.load(f))
    return qs


def run_board(split=SPLIT):
    print("=" * 60)
    print(f"AFAC2026 {split}榜 — {'B榜自动检索' if split == 'B' else 'A榜(有doc_ids)'}")
    print("=" * 60)

    questions = load_questions(split)
    if not questions:
        print(f"无 {split} 榜题目. B 榜 7/22 开放.")
        return
    print(f"加载 {len(questions)} 题")

    di = DocumentIndex(); di.load()

    # B 榜: 构建文档检索器 (A 榜不需要, resolve_doc_ids 会直接返回 doc_ids)
    retriever = DocRetriever(di) if split == "B" else None
    if retriever:
        print(f"  B 榜检索就绪 (纯 BM25, 零 token). top-k={B_RETRIEVE_TOPK}")

    qwen = QwenClient()
    agent = ReasoningAgentV31(qwen, di, None, token_budget=TOKEN_BUDGET)

    print("\n开始推理...\n")
    results = []
    n_retrieved = 0
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print("⚠️ token 上限")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break

        # A/B 自适应: 有 doc_ids 用原值, 无则检索
        if retriever and not q.get("doc_ids"):
            doc_ids = retriever.resolve_doc_ids(q, top_k=B_RETRIEVE_TOPK,
                                                use_vector=USE_VECTOR)
            q = {**q, "doc_ids": doc_ids}  # 注入检索到的 doc_ids
            n_retrieved += 1

        print(f"[{i+1}/{len(questions)}]", end="")
        r = agent.answer_question(q)
        results.append({"qid": q["qid"], "answer": r["answer"]})
        print(f' {q["qid"]} → {r["answer"]} [证据{r["evidence_chars"]//1000}K]')

    if n_retrieved:
        print(f"\n  B 榜: {n_retrieved} 题自动检索了候选文档")

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total = stats["total_tokens"]
    ts = max(0, min(1, (TOKEN_BUDGET - total) / TOKEN_BUDGET))

    suffix = f"_{split.lower()}" if split != "A" else ""
    out = generate_answer_csv_token_stats(
        results, stats["prompt_tokens"], stats["completion_tokens"], total)
    if suffix:
        import shutil
        shutil.copy(out, os.path.join(RESULTS_DIR, f"answer{suffix}.csv"))

    agent.save_cot_trails()

    ans = [r["answer"] for r in results if r["answer"]]
    dist = Counter(ans).most_common(15)
    single = [a for a in ans if len(a) == 1]
    sd = {c: single.count(c) for c in "ABCD"}

    print(f"\n📊 {split}榜摘要:")
    print(f"  有效: {sum(1 for r in results if r['answer'])}/{len(questions)}")
    print(f"  Token: {total:,}")
    print(f"  TokenScore: {ts:.4f}")
    print(f"  调用: {stats['call_count']}")
    print(f"  分布: {dist}")
    print(f"  单选: A={sd.get('A',0)} B={sd.get('B',0)} C={sd.get('C',0)} D={sd.get('D',0)}")


if __name__ == "__main__":
    run_board()
