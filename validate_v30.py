"""V30 LLM 验证: 10 题, 看 V30 用精炼证据(8-14K)是否仍能答好

V22 相同题的答案作为参照.
"""
import os, sys, json, time, csv
sys.path.insert(0, ".")
from agent.config import QUESTIONS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v30 import ReasoningAgentV30

SAMPLES = [
    ("fin_a_001", "fin multi", "AB"),  # BYD 财报, V22=AB
    ("fin_a_004", "fin multi", "ABD"), # Midea 财报, V22=ABD
    ("ins_a_007", "ins multi", "C"),   # 保单贷款, V22=C
    ("ins_a_002", "ins mcq",   "A"),   # 退保计算, V22=A
    ("fc_a_001",  "fc multi",  "ABD"), # 合同比较, V22=ABD
    ("fc_a_014",  "fc multi",  "C"),   # 合同条款, V22=C
    ("reg_a_010", "reg tf",    "B"),   # 大额vs可疑, V22=B
    ("res_a_006", "res tf",    "B"),   # 研报数字, V22=B
    ("res_a_001", "res multi", "ABC"), # 研报多选, V22=ABC
    ("ins_a_019", "ins multi", "ACD"),  # 保险多选, V22=ACD
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
    agent = ReasoningAgentV30(qwen, di, None)

    os.makedirs("dry_run_outputs/v30_validate", exist_ok=True)
    results = []

    for qid, desc, v22_ans in SAMPLES:
        q = qmap[qid]
        print(f"\n[{qid}] {desc}  V22={v22_ans}")
        t0 = time.time()
        r = agent.answer_question(q)
        elapsed = time.time() - t0
        v30 = r["answer"]
        raw = agent.cot_trails[-1]["raw_response"]
        flag = "★" if v30 != v22_ans else " "
        print(f"  V30={v30}  ({elapsed:.1f}s) ev={r['evidence_chars']//1000}K {flag}")

        with open(f"dry_run_outputs/v30_validate/{qid}.txt", "w", encoding="utf-8") as f:
            f.write(f"V22={v22_ans} | V30={v30}\n\n=== Raw ({len(raw)} chars) ===\n{raw}")

        results.append({"qid": qid, "v22": v22_ans, "v30": v30, "ev_K": r["evidence_chars"]//1000})

    stats = qwen.get_token_stats()
    print(f"\n=== V30 验证总结 ===")
    print(f"Token: {stats['total_tokens']:,}")
    print()
    print(f'{"qid":12s} {"V22":6s} {"V30":6s} {"ev_K":>5s}')
    for r in results:
        flag = "★" if r["v30"] != r["v22"] else " "
        print(f"  {r['qid']:12s} {r['v22']:6s} {r['v30']:6s} {r['ev_K']:>5d}  {flag}")


if __name__ == "__main__":
    main()
