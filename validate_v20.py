"""V20 LLM 验证: 在 5 道"V13 关键数字 0 次出现"的题上调 LLM, 看 V20 是否真能答好.

成本 ~150K token ~¥1, 但是这次验证有强意义:
- 不是看 prompt 看着合理, 是看模型在有数据时真的会用
- V13 在这些题上必错 (无数据), V20 至少有数据
"""
import os, sys, json, re, time
sys.path.insert(0, ".")
from agent.config import QUESTIONS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v20 import ReasoningAgentV20

# 关键题: 都是 V13 已经在做 (有 V13 答案对比), 都是真实数字精确比对
SAMPLES = [
    ("fin_a_001", "比亚迪 24/25 财报对比 multi"),
    ("fin_a_002", "宁德 24/25 财报对比 multi"),
    ("fin_a_004", "中国移动财报 multi"),
    ("fc_a_001",  "广晟控股 vs 第二份 合同 multi"),
    ("ins_a_002", "保险计算题 mcq"),
    ("reg_a_010", "tf 复合陈述大额/可疑"),
    ("res_a_001", "研报数字 56% 9.9% 2500亿 multi"),
]

V13_ANSWERS = {
    "fin_a_001": "AB",
    "fin_a_002": "B",
    "fin_a_004": "D",
    "fc_a_001": "A",
    "ins_a_002": "A",
    "reg_a_010": "A",
    "res_a_001": "AB",
}


def load_qmap():
    qs = []
    for fn in sorted(os.listdir(os.path.join(QUESTIONS_DIR, "group_a"))):
        if fn.endswith(".json"):
            with open(os.path.join(QUESTIONS_DIR, "group_a", fn), encoding="utf-8") as f:
                qs.extend(json.load(f))
    return {q["qid"]: q for q in qs}


def main():
    print("Loading ...")
    di = DocumentIndex(); di.load()
    qmap = load_qmap()
    qwen = QwenClient()
    agent = ReasoningAgentV20(qwen, di, None)

    out_dir = "dry_run_outputs/v20_validate"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for qid, why in SAMPLES:
        q = qmap[qid]
        print(f"\n[{qid}] {why}")
        t0 = time.time()
        result = agent.answer_question(q)
        elapsed = time.time() - t0
        v20 = result["answer"]
        v13 = V13_ANSWERS.get(qid, "?")
        raw = agent.cot_trails[-1]["raw_response"]

        flag = "★" if v20 != v13 else " "
        print(f"  V20={v20} | V13={v13} | {elapsed:.1f}s | {flag}")

        with open(f"{out_dir}/{qid}.txt", "w", encoding="utf-8") as f:
            f.write(f"V20={v20} | V13={v13}\n\n=== Raw Response ({len(raw)} chars) ===\n{raw}")

        results.append({"qid": qid, "v13": v13, "v20": v20, "flag": flag, "raw_chars": len(raw)})

    stats = qwen.get_token_stats()
    print(f"\n=== 总结 ===")
    print(f"Token: {stats['total_tokens']:,}")
    print(f"调用: {stats['call_count']} 次")
    changed = sum(1 for r in results if r["flag"] == "★")
    print(f"V20 vs V13 答案变化: {changed}/{len(results)}")
    print()
    print(f'{"qid":12s} {"V13":5s} {"V20":5s} {"raw chars":10s}')
    for r in results:
        print(f"  {r['qid']:12s} {r['v13']:5s} {r['v20']:5s} {r['raw_chars']:10d}")


if __name__ == "__main__":
    main()
