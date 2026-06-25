"""V34 全量流水线: V31 架构 + enable_thinking=False"""
import os, json
from collections import Counter
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v34 import ReasoningAgentV34
from agent.postprocessor import generate_answer_csv_token_stats


def load_questions(split="A"):
    qs = []
    qa = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    if not os.path.exists(qa):
        return qs
    for fn in sorted(os.listdir(qa)):
        if fn.endswith(".json"):
            with open(os.path.join(qa, fn), encoding="utf-8") as f:
                qs.extend(json.load(f))
    return qs


def run_a_board():
    print("=" * 60)
    print("AFAC2026 V34 — V31 + enable_thinking=False")
    print("=" * 60)

    questions = load_questions("A")
    print(f"加载 {len(questions)} 题")

    di = DocumentIndex(); di.load()
    qwen = QwenClient()
    agent = ReasoningAgentV34(qwen, di, None, token_budget=TOKEN_BUDGET)

    print("\n开始推理...\n")
    results = []
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print("token 上限")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break
        fmt = q.get("answer_format", "mcq")
        print(f"[{i+1}/{len(questions)}]", end="")
        r = agent.answer_question(q)
        results.append({"qid": q["qid"], "answer": r["answer"]})
        print(f' {q["qid"]} ({q.get("domain","")[:4]}/{fmt}) -> {r["answer"]} [ev{r["evidence_chars"]//1000}K]')

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total = stats["total_tokens"]
    ts = max(0, min(1, (TOKEN_BUDGET - total) / TOKEN_BUDGET))

    out = generate_answer_csv_token_stats(
        results, stats["prompt_tokens"], stats["completion_tokens"], total)

    import shutil
    shutil.copy(out, os.path.join(RESULTS_DIR, "answer_v34.csv"))
    print(f"  备份 V34 -> {os.path.join(RESULTS_DIR, 'answer_v34.csv')}")

    agent.save_cot_trails()

    ans = [r["answer"] for r in results if r["answer"]]
    dist = Counter(ans).most_common(15)
    multi_single = sum(1 for r in results
                       if len(r["answer"]) == 1 and
                       any(q["qid"] == r["qid"] and q.get("answer_format") == "multi"
                           for q in questions))

    print(f"\nV34 摘要:")
    print(f"  有效: {sum(1 for r in results if r['answer'])}/{len(questions)}")
    print(f"  Token: {total:,}")
    print(f"  TokenScore: {ts:.4f}")
    print(f"  调用: {stats['call_count']}")
    print(f"  分布: {dist}")
    print(f"  multi单字母: {multi_single}")
    print(f"\n  参照:")
    print(f"    V31: Token=3.19M, TS=0.362, Score=48.50")
    print(f"    V34预测: Token~2.23M, TS~0.55, Score~52.0 (+3.5)")


if __name__ == "__main__":
    run_a_board()
