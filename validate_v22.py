"""V22 LLM 验证: 选 V20 答错可能性高的关键题, 看 V22 是否真改对

关键题选择依据 (V20 raw 诊断):
- fc 域: fc_a_007/008/009/012/014 (推理质量低, uncertain ≥6)
- fin 域: fin_a_004/007/014 (高 uncertain 或 no_evidence)
- 已验证 7 题中: ins_a_007 V20=A 可能错 (我离线判 BC), 也加进来

共 10 题, ~250K token = ~¥1.5
"""
import os, sys, json, time
sys.path.insert(0, ".")
from agent.config import QUESTIONS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v22 import ReasoningAgentV22

SAMPLES = [
    "fc_a_007", "fc_a_008", "fc_a_009", "fc_a_012", "fc_a_014",
    "fin_a_004", "fin_a_007", "fin_a_014",
    "ins_a_007", "res_a_011",  # ins_a_007 V20=A 可能错(我判 BC); res_a_011 fc 旁边
]


def load_qmap():
    qs = []
    for fn in sorted(os.listdir(os.path.join(QUESTIONS_DIR, "group_a"))):
        if fn.endswith(".json"):
            with open(os.path.join(QUESTIONS_DIR, "group_a", fn), encoding="utf-8") as f:
                qs.extend(json.load(f))
    return {q["qid"]: q for q in qs}


# V20 答案
V20_ANS = {}
import csv
with open("results/answer_v20.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["qid"] != "summary":
            V20_ANS[r["qid"]] = r["answer"]


def main():
    print("Loading ...")
    di = DocumentIndex(); di.load()
    qmap = load_qmap()
    qwen = QwenClient()
    agent = ReasoningAgentV22(qwen, di, None)

    out_dir = "dry_run_outputs/v22_validate"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for qid in SAMPLES:
        q = qmap[qid]
        v20 = V20_ANS.get(qid, "?")
        print(f"\n[{qid}] {q['domain'][:10]}/{q['answer_format']}  V20={v20}")
        t0 = time.time()
        r = agent.answer_question(q)
        elapsed = time.time() - t0
        v22 = r["answer"]
        raw = agent.cot_trails[-1]["raw_response"]
        flag = "★" if v22 != v20 else " "
        print(f"  V22={v22}  ({elapsed:.1f}s) ev={r['evidence_chars']//1000}K {flag}")

        with open(f"{out_dir}/{qid}.txt", "w", encoding="utf-8") as f:
            f.write(f"V20={v20} | V22={v22}\n\n=== Raw ({len(raw)} chars) ===\n{raw}")

        results.append({"qid": qid, "v20": v20, "v22": v22, "ev_chars": r["evidence_chars"], "flag": flag})

    stats = qwen.get_token_stats()
    print(f"\n=== 总结 ===")
    print(f"Token: {stats['total_tokens']:,}")
    changed = sum(1 for r in results if r["flag"] == "★")
    print(f"V22 vs V20 答案变化: {changed}/{len(results)}\n")
    print(f'{"qid":12s} {"V20":6s} {"V22":6s} {"ev_K":>5s}')
    for r in results:
        print(f"  {r['qid']:12s} {r['v20']:6s} {r['v22']:6s} {r['ev_chars']//1000:>5d}")


if __name__ == "__main__":
    main()
