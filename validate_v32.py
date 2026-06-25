"""V32 验证: 6 题 multi，观察 ReAct 循环行为和记忆压缩质量"""
import os, sys, json, time
sys.path.insert(0, ".")
from agent.config import QUESTIONS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v32 import ReasoningAgentV32

SAMPLES = [
    ("ins_a_007", "ins multi", "C"),    # 保单贷款(跨文档否定推理)
    ("ins_a_019", "ins multi", "ACD"),  # 保险多选
    ("fin_a_001", "fin multi", "AB"),   # 财报多选
    ("fc_a_001",  "fc multi",  "ABD"),  # 合同多选
    ("reg_a_001", "reg multi", "ACD"),  # 法规多选
    ("res_a_001", "res multi", "ABC"),  # 研报多选
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
    agent = ReasoningAgentV32(qwen, di, None)

    os.makedirs("dry_run_outputs/v32_validate", exist_ok=True)
    results = []

    for qid, desc, v31_ans in SAMPLES:
        q = qmap[qid]
        print(f"\n{'='*50}")
        print(f"[{qid}] {desc}  V31={v31_ans}")
        t0 = time.time()
        r = agent.answer_multi_react_verbose(q) if hasattr(agent, 'answer_multi_react_verbose') else agent.answer_question(q)
        elapsed = time.time() - t0
        v32 = r["answer"]
        flag = "SAME" if v32 == v31_ans else "DIFF"
        print(f"  V32={v32}  ({elapsed:.1f}s) ev={r['evidence_chars']//1000}K [{flag}]")

        # 保存完整 raw_response
        trail = agent.cot_trails[-1]
        raw = trail.get("raw_response", "")
        with open(f"dry_run_outputs/v32_validate/{qid}.txt", "w", encoding="utf-8") as f:
            f.write(f"V31={v31_ans} | V32={v32}\n\n{raw}")

        results.append({"qid": qid, "v31": v31_ans, "v32": v32,
                        "ev_K": r["evidence_chars"]//1000, "elapsed": elapsed})

    stats = qwen.get_token_stats()
    print(f"\n{'='*50}")
    print(f"=== V32 验证总结 ===")
    print(f"Token: {stats['total_tokens']:,}  (6题, 平均{stats['total_tokens']//6:,}/题)")
    print(f"调用数: {stats['call_count']}")
    print()
    print(f'{"qid":12s} {"V31":6s} {"V32":6s} {"ev_K":>5s} {"t":>5s}')
    for r in results:
        flag = " " if r["v32"] == r["v31"] else "*"
        print(f"  {r['qid']:12s} {r['v31']:6s} {r['v32']:6s} {r['ev_K']:>5d}  {r['elapsed']:>4.1f}s  {flag}")

    same = sum(1 for r in results if r["v32"] == r["v31"])
    print(f"\n与 V31 相同: {same}/{len(results)}")
    print(f"平均每题 token: {stats['total_tokens']//len(results):,}")


if __name__ == "__main__":
    main()
