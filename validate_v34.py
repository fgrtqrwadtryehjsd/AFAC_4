"""V34 验证: 先跑 5 题（覆盖 multi + tf/mcq），确认答案与 V31 一致且 token 显著降低"""
import os, sys, json, time
sys.path.insert(0, ".")
from agent.config import QUESTIONS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v34 import ReasoningAgentV34

SAMPLES = [
    ("ins_a_007",  "multi", "C"),
    ("fc_a_001",   "multi", "ABD"),
    ("reg_a_001",  "multi", "ACD"),
    ("fc_a_003",   "tf",    "B"),
    ("ins_a_001",  "mcq",   "B"),
]

def load_qmap():
    qs = []
    for fn in sorted(os.listdir(os.path.join(QUESTIONS_DIR, "group_a"))):
        if fn.endswith(".json"):
            with open(os.path.join(QUESTIONS_DIR, "group_a", fn), encoding="utf-8") as f:
                qs.extend(json.load(f))
    return {q["qid"]: q for q in qs}

def main():
    di = DocumentIndex(); di.load()
    qmap = load_qmap()
    qwen = QwenClient()
    agent = ReasoningAgentV34(qwen, di, None)

    results = []
    for qid, fmt, v31_ans in SAMPLES:
        q = qmap[qid]
        print(f"\n[{qid}] {fmt}  V31={v31_ans}")
        t0 = time.time()
        r = agent.answer_question(q)
        elapsed = time.time() - t0
        v34 = r["answer"]
        flag = "SAME" if v34 == v31_ans else "DIFF"
        print(f"  V34={v34}  ({elapsed:.1f}s) ev={r['evidence_chars']//1000}K [{flag}]")
        results.append({"qid": qid, "v31": v31_ans, "v34": v34, "elapsed": elapsed})

    stats = qwen.get_token_stats()
    print(f"\n{'='*50}")
    print(f"Token: {stats['total_tokens']:,} ({len(results)}题, avg={stats['total_tokens']//len(results):,}/题)")
    print(f"调用数: {stats['call_count']}")
    same = sum(1 for r in results if r["v34"] == r["v31"])
    print(f"与 V31 相同: {same}/{len(results)}")
    print()

    # 推算全量 token
    est_total = stats['total_tokens'] / len(results) * 100
    ts = (5_000_000 - est_total) / 5_000_000
    print(f"推算全量 token: {est_total/1e6:.2f}M")
    print(f"推算 TS: {ts:.3f}")
    print(f"推算 Score (Acc=60%): {60 * (0.7 + 0.3 * ts):.1f}")

if __name__ == "__main__":
    main()
