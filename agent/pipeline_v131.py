"""V13.1 流水线 — V13 + doc 级配额 + 同义词扩展

只改 retriever, 其他全部与 V13 相同.
"""
import os
import json
from collections import Counter
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.vector_indexer import VectorIndexer
from agent.reasoner_v131 import ReasoningAgentV131
from agent.postprocessor import generate_answer_csv_token_stats


def load_questions(split: str = "A") -> list:
    questions = []
    questions_dir = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    if not os.path.exists(questions_dir):
        return questions
    for filename in sorted(os.listdir(questions_dir)):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(questions_dir, filename), "r", encoding="utf-8") as f:
            questions.extend(json.load(f))
    return questions


def run_a_board():
    print("=" * 60)
    print("AFAC2026 赛题四 - A 榜评测 V13.1")
    print("V13 + doc 级配额 + 同义词扩展 (只改 retriever)")
    print(f"模型: qwen-plus | Token 预算: {TOKEN_BUDGET:,}")
    print("=" * 60)

    questions = load_questions("A")
    print(f"加载了 {len(questions)} 道 A 榜题目")

    print("\n🔍 构建检索索引...")
    doc_index = DocumentIndex()
    doc_index.load()

    print("\n🔮 构建语义向量索引...")
    vector_indexer = VectorIndexer(doc_index)

    qwen = QwenClient()
    agent = ReasoningAgentV131(qwen, doc_index, vector_indexer, token_budget=TOKEN_BUDGET)

    print(f"\n🧠 开始推理 (融合证据+RRF3路+doc配额+同义词)...")
    print("=" * 60)

    results = []
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print(f"\n⚠️ Token 接近上限 ({stats['total_tokens']:,})")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break

        print(f"[{i+1}/{len(questions)}]", end="")
        result = agent.answer_question(q)
        answer = result["answer"]
        results.append({"qid": q["qid"], "answer": answer})

        domain = q.get("domain", "")
        fmt = q.get("answer_format", "")
        ev_chars = result.get("evidence_chars", 0)
        total_doc = result.get("total_doc_chars", 0)
        is_full = total_doc <= agent.FULL_DOC_THRESHOLD
        tag = "全文" if is_full else "融合检索"
        print(f" {q['qid']} ({domain}/{fmt}) → {answer} "
              f"[{tag}:证据{ev_chars//1000}K/{total_doc//1000}K]")

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total_tokens = stats["total_tokens"]
    token_score = max(0, min(1, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET))
    valid = sum(1 for r in results if r["answer"])

    output_path = generate_answer_csv_token_stats(
        results, stats["prompt_tokens"], stats["completion_tokens"], total_tokens)

    # 另存一份以版本命名,避免覆盖 V13 的 answer.csv
    v131_csv = os.path.join(RESULTS_DIR, "answer_v131.csv")
    import shutil
    shutil.copy(output_path, v131_csv)
    print(f"  备份 V13.1 → {v131_csv}")

    agent.save_cot_trails()

    if vector_indexer:
        vector_indexer.finalize()
        cache_stats = vector_indexer.get_cache_stats()
        print(f"  Embedding缓存: API{cache_stats['api_calls']}次, "
              f"命中{cache_stats['cache_hits']}次")

    answers = [r["answer"] for r in results if r["answer"]]
    answer_dist = Counter(answers).most_common(15)
    single = [a for a in answers if len(a) == 1]
    single_dist = {c: single.count(c) for c in "ABCD"}

    print(f"\n📊 V13.1 评测摘要:")
    print(f"  有效答案: {valid}/{len(questions)}")
    print(f"  总 Token: {total_tokens:,}")
    print(f"  TokenScore: {token_score:.4f}")
    print(f"  API调用: {stats['call_count']}次")
    print(f"\n  答案分布: {answer_dist}")
    print(f"  单选分布: A={single_dist.get('A',0)} B={single_dist.get('B',0)} "
          f"C={single_dist.get('C',0)} D={single_dist.get('D',0)}")

    print(f"\n  📊 V13参照: A=39 B=13 C=7 D=5 | TokenScore=0.630 | Score=39.12")
    print(f"  ✅ 结果: {output_path} (备份: {v131_csv})")


if __name__ == "__main__":
    run_a_board()
