"""V44 流水线: V44(multi: V35+靶向grep+claim核验draft+claim核验审计) + V43(tf/mcq)

multi: V44 = V35广域 + 靶向grep(补深位数) + PROMPT_MULTI_V44 claim核验 + V44审计claim核验
tf/mcq: V43 靶向证据+宽容prompt(不变)
"""
import os, json, shutil
from collections import Counter
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v44 import ReasoningAgentV44
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
    print("AFAC2026 V44 — multi claim核验+靶向grep; tf/mcq V43 (合规, 仅Qwen)")
    print("=" * 60)

    questions = load_questions("A")
    print(f"加载 {len(questions)} 题")

    di = DocumentIndex(); di.load()
    qwen = QwenClient()
    agent = ReasoningAgentV44(qwen, di, None, token_budget=TOKEN_BUDGET)

    print("\n开始推理...\n")
    results = []
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print("⚠️ token 上限")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break
        print(f"[{i+1}/{len(questions)}]", end="")
        r = agent.answer_question(q)
        results.append({"qid": q["qid"], "answer": r["answer"]})
        af = q.get("answer_format", "")
        strat = "v44multi" if af == "multi" else "v43tfmq"
        print(f' {q["qid"]} ({q.get("domain","")[:4]}/{af}) → {r["answer"]} [{strat}|ev{r["evidence_chars"]//1000}K/{r["total_doc_chars"]//1000}K]')
        if (i + 1) % 20 == 0:
            st = qwen.get_token_stats()
            print(f"   ...累计 token {st['total_tokens']:,}")

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total = stats["total_tokens"]
    ts = max(0, min(1, (TOKEN_BUDGET - total) / TOKEN_BUDGET))
    factor = 0.7 + 0.3 * ts

    out = generate_answer_csv_token_stats(
        results, stats["prompt_tokens"], stats["completion_tokens"], total)

    v44_csv = os.path.join(RESULTS_DIR, "answer_v44.csv")
    shutil.copy(out, v44_csv)
    print(f"  备份 V44 → {v44_csv}")

    agent.save_cot_trails(path=os.path.join(RESULTS_DIR, "eval_results_v44.json"))

    ans = [r["answer"] for r in results if r["answer"]]
    dist = Counter(ans).most_common(15)
    single = [a for a in ans if len(a) == 1]
    sd = {c: single.count(c) for c in "ABCD"}

    print(f"\n📊 V44 摘要:")
    print(f"  有效: {sum(1 for r in results if r['answer'])}/{len(questions)}")
    print(f"  Token: {total:,}")
    print(f"  TokenScore: {ts:.4f}  (factor={factor:.4f})")
    print(f"  调用: {stats['call_count']}")
    print(f"  分布: {dist}")
    print(f"  单选: A={sd.get('A',0)} B={sd.get('B',0)} C={sd.get('C',0)} D={sd.get('D',0)}")
    print(f"\n  📊 参照: V43=66.79(84对,3.42M) | test_A=74.38(92对,不合规)")
    for n in (88, 89, 90, 91, 92):
        print(f"  V44 若 {n}对: Score={n}×{factor:.3f}={n*factor:.1f}")
    print(f"  ✅ 结果: {out}")


if __name__ == "__main__":
    run_a_board()
