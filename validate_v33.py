"""V33 验证: 每个 domain 各取 1-2 道 multi 题"""
import os, sys, json, time
sys.path.insert(0, ".")
from agent.config import QUESTIONS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v33 import ReasoningAgentV33

# V31 参考答案（已知正确）
SAMPLES = [
    ("ins_a_007", "insurance",   "C"),
    ("ins_a_005", "insurance",   "ABD"),
    ("fc_a_001",  "financial_contracts", "ABD"),
    ("fc_a_004",  "financial_contracts", "ACD"),
    ("fin_a_001", "financial_reports",   "AB"),
    ("fin_a_002", "financial_reports",   "ABD"),
    ("reg_a_001", "regulatory",  "ACD"),
    ("reg_a_002", "regulatory",  "ABC"),
    ("res_a_001", "research",    "ABC"),
    ("res_a_002", "research",    "AC"),
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
    agent = ReasoningAgentV33(qwen, di, None)

    os.makedirs("dry_run_outputs/v33_validate", exist_ok=True)
    results = []

    for qid, domain, v31_ans in SAMPLES:
        q = qmap[qid]
        print(f"\n{'='*50}")
        print(f"[{qid}] {domain[:4]}  V31={v31_ans}")
        t0 = time.time()
        r = agent.answer_question(q)
        elapsed = time.time() - t0
        v33 = r["answer"]
        flag = "SAME" if v33 == v31_ans else "DIFF"
        print(f"  V33={v33}  ({elapsed:.1f}s) ev={r['evidence_chars']//1000}K [{flag}]")

        trail = agent.cot_trails[-1]
        raw = trail.get("raw_response", "")
        with open(f"dry_run_outputs/v33_validate/{qid}.txt", "w", encoding="utf-8") as f:
            f.write(f"V31={v31_ans} | V33={v33}\n\n{raw}")

        results.append({"qid": qid, "v31": v31_ans, "v33": v33,
                        "ev_K": r["evidence_chars"]//1000, "elapsed": elapsed})

    stats = qwen.get_token_stats()
    print(f"\n{'='*50}")
    print(f"=== V33 验证总结 ===")
    print(f"Token: {stats['total_tokens']:,}  ({len(results)}题, 平均{stats['total_tokens']//len(results):,}/题)")
    print(f"调用数: {stats['call_count']}")
    print()
    print(f'{"qid":15s} {"V31":6s} {"V33":6s} {"ev_K":>5s} {"t":>5s}')
    for r in results:
        flag = " " if r["v33"] == r["v31"] else "*"
        print(f"  {r['qid']:15s} {r['v31']:6s} {r['v33']:6s} {r['ev_K']:>5d}  {r['elapsed']:>4.1f}s  {flag}")

    same = sum(1 for r in results if r["v33"] == r["v31"])
    print(f"\n与 V31 相同: {same}/{len(results)}")
    print(f"平均每题 token: {stats['total_tokens']//len(results):,}")

if __name__ == "__main__":
    main()
