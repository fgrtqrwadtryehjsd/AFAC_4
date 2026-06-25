"""V31 快速验证: 10 题, 确认 multi 不再单字母, tf/mcq 仍精炼"""
import os, sys, json, time
sys.path.insert(0, ".")
from agent.config import QUESTIONS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v31 import ReasoningAgentV31

# 覆盖各域各类型
SAMPLES = [
    ("fin_a_001", "fin multi",  "AB"),   # 财报 multi
    ("fin_a_004", "fin multi",  "ABD"),  # 财报 multi
    ("ins_a_007", "ins multi",  "C"),    # 保险 multi
    ("ins_a_019", "ins multi",  "ACD"),  # 保险 multi
    ("res_a_001", "res multi",  "ABC"),  # 研报 multi
    ("fc_a_001",  "fc multi",   "ABD"),  # 合同 multi
    ("reg_a_010", "reg tf",     "B"),    # 法规 tf
    ("ins_a_002", "ins mcq",    "A"),    # 保险 mcq
    ("res_a_006", "res tf",     "B"),    # 研报 tf
    ("fc_a_014",  "fc multi",   "C"),    # 合同 multi
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
    agent = ReasoningAgentV31(qwen, di, None)

    os.makedirs("dry_run_outputs/v31_validate", exist_ok=True)
    results = []
    multi_single = 0

    for qid, desc, v22_ans in SAMPLES:
        q = qmap[qid]
        fmt = q.get("answer_format", "mcq")
        print(f"\n[{qid}] {desc} (fmt={fmt})  V22={v22_ans}")
        t0 = time.time()
        r = agent.answer_question(q)
        elapsed = time.time() - t0
        v31 = r["answer"]
        flag = "★" if v31 != v22_ans else " "
        if fmt == "multi" and len(v31) == 1:
            multi_single += 1
            flag += " [单字母!]"
        print(f"  V31={v31}  ({elapsed:.1f}s) ev={r['evidence_chars']//1000}K {flag}")

        raw = agent.cot_trails[-1]["raw_response"]
        with open(f"dry_run_outputs/v31_validate/{qid}.txt", "w", encoding="utf-8") as f:
            f.write(f"V22={v22_ans} | V31={v31}\n\n=== Raw ({len(raw)} chars) ===\n{raw}")

        results.append({"qid": qid, "v22": v22_ans, "v31": v31,
                        "fmt": fmt, "ev_K": r["evidence_chars"]//1000})

    stats = qwen.get_token_stats()
    print(f"\n=== V31 验证总结 ===")
    print(f"Token: {stats['total_tokens']:,}")
    print(f"multi 单字母: {multi_single}/6")
    print()
    print(f'{"qid":12s} {"fmt":6s} {"V22":6s} {"V31":6s} {"ev_K":>5s}')
    for r in results:
        flag = "★" if r["v31"] != r["v22"] else " "
        if r["fmt"] == "multi" and len(r["v31"]) == 1:
            flag += "[单!]"
        print(f"  {r['qid']:12s} {r['fmt']:6s} {r['v22']:6s} {r['v31']:6s} {r['ev_K']:>5d}  {flag}")


if __name__ == "__main__":
    main()
