"""V30 流水线"""
import os, json
from collections import Counter
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v30 import ReasoningAgentV30
from agent.postprocessor import generate_answer_csv_token_stats


def load_questions(split="A"):
    qs = []
    qa = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    if not os.path.exists(qa): return qs
    for fn in sorted(os.listdir(qa)):
        if fn.endswith(".json"):
            with open(os.path.join(qa, fn), encoding="utf-8") as f:
                qs.extend(json.load(f))
    return qs


def run_a_board():
    print("=" * 60)
    print("AFAC2026 V30 — 结构化预提取 + 大幅降 Token")
    print("=" * 60)

    questions = load_questions("A")
    print(f"加载 {len(questions)} 题")

    di = DocumentIndex(); di.load()
    qwen = QwenClient()
    agent = ReasoningAgentV30(qwen, di, None, token_budget=TOKEN_BUDGET)

    print("\n开始推理 (结构化证据)...\n")
    results = []
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print("⚠️ token 上限")
            for rq in questions[i:]: results.append({"qid": rq["qid"], "answer": ""})
            break
        print(f"[{i+1}/{len(questions)}]", end="")
        r = agent.answer_question(q)
        results.append({"qid": q["qid"], "answer": r["answer"]})
        print(f' {q["qid"]} ({q.get("domain","")[:4]}/{q.get("answer_format","")}) → {r["answer"]} [证据{r["evidence_chars"]//1000}K/{r["total_doc_chars"]//1000}K]')

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total = stats["total_tokens"]
    ts = max(0, min(1, (TOKEN_BUDGET - total) / TOKEN_BUDGET))

    out = generate_answer_csv_token_stats(
        results, stats["prompt_tokens"], stats["completion_tokens"], total)

    import shutil
    v30_csv = os.path.join(RESULTS_DIR, "answer_v30.csv")
    shutil.copy(out, v30_csv)
    print(f"  备份 V30 → {v30_csv}")

    agent.save_cot_trails()

    ans = [r["answer"] for r in results if r["answer"]]
    dist = Counter(ans).most_common(15)
    single = [a for a in ans if len(a)==1]
    sd = {c: single.count(c) for c in "ABCD"}

    print(f"\n📊 V30 摘要:")
    print(f"  有效: {sum(1 for r in results if r['answer'])}/{len(questions)}")
    print(f"  Token: {total:,}")
    print(f"  TokenScore: {ts:.4f}")
    print(f"  调用: {stats['call_count']}")
    print(f"  分布: {dist}")
    print(f"  单选: A={sd.get('A',0)} B={sd.get('B',0)} C={sd.get('C',0)} D={sd.get('D',0)}")
    print(f"\n  📊 V22 参照: Token=4.25M, TS=0.149, Score=44.69")
    print(f"  预期 V30: Token~0.56M, TS~0.89, 若Acc≥57%→Score≥55")


if __name__ == "__main__":
    run_a_board()
