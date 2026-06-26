"""V16 流水线 — Qwen-Long 原生文件上传 + 混合策略

V16 架构：
  - 长文档 (>60K chars)：qwen-long 文件上传，1000万 Token 上下文全文推理
  - 短文档 (≤60K chars)：qwen-plus 全文内嵌（V15策略）
  - 混合 Token 预算管理：qwen-long + qwen-plus 联合计算

Token 分析：
  - qwen-long：0.0005元/千Token（比 qwen-plus 便宜约10倍）
  - 5M Token 预算下，qwen-long 可处理约 50 份 100K 文档
  - 预期：100题总 Token ≤ 3M，TokenScore ≥ 0.4

预期得分：
  - qwen-long 全文推理消除检索遗漏 → Accuracy 从 44% 提升到 65%+
  - FinalScore = 100 × 0.65 × (0.7 + 0.3 × 0.4) ≈ 54分
"""

import os
import json
from collections import Counter
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.qwen_long_client import QwenLongClient
from agent.indexer import DocumentIndex
from agent.vector_indexer import VectorIndexer
from agent.reasoner_v16 import ReasoningAgentV16
from agent.postprocessor import generate_answer_csv_token_stats


def load_questions(split: str = "A") -> list:
    questions = []
    questions_dir = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    if not os.path.exists(questions_dir):
        return questions
    for filename in sorted(os.listdir(questions_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(questions_dir, filename), "r", encoding="utf-8") as f:
            questions.extend(json.load(f))
    return questions


def run_a_board():
    print("=" * 60)
    print("AFAC2026 赛题四 - A 榜评测 V16")
    print("Qwen-Long 文件上传 + 混合策略（长文档全文/短文档检索）")
    print(f"Token 预算: {TOKEN_BUDGET:,}")
    print("=" * 60)

    questions = load_questions("A")
    print(f"加载了 {len(questions)} 道 A 榜题目")

    print("\n🔍 构建检索索引（用于短文档回退策略）...")
    doc_index = DocumentIndex()
    doc_index.load()

    print("\n🔮 构建语义向量索引...")
    vector_indexer = VectorIndexer(doc_index)

    qwen = QwenClient()
    qwen_long = QwenLongClient()

    agent = ReasoningAgentV16(
        qwen=qwen,
        qwen_long=qwen_long,
        doc_index=doc_index,
        vector_indexer=vector_indexer,
        token_budget=TOKEN_BUDGET,
    )

    print("\n📤 预上传长文档到 qwen-long...")
    agent.preupload_documents(questions)

    print(f"\n🧠 开始推理...")
    print("=" * 60)

    results = []
    for i, q in enumerate(questions):
        # Token 预算：qwen-plus + qwen-long 合计
        plus_tokens = qwen.get_token_stats()["total_tokens"]
        long_tokens = qwen_long.get_token_stats()["total_tokens"]
        total_used = plus_tokens + long_tokens

        if total_used > TOKEN_BUDGET * 0.95:
            print(f"\n⚠️ Token 接近上限 ({total_used:,})")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break

        print(f"[{i+1}/{len(questions)}]", end="")
        result = agent.answer_question(q)
        answer = result["answer"]
        results.append({"qid": q["qid"], "answer": answer})

        domain = q.get("domain", "")
        fmt = q.get("answer_format", "")
        ev_chars = result.get("evidence_chars", 0)
        total_doc = result.get("total_doc_chars", 0)
        strategy = result.get("strategy", "v15")
        print(
            f" {q['qid']} ({domain}/{fmt}) → {answer} "
            f"[{strategy}|{ev_chars//1000}K/{total_doc//1000}K]"
        )

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plus_stats = qwen.get_token_stats()
    long_stats = qwen_long.get_token_stats()
    total_tokens = plus_stats["total_tokens"] + long_stats["total_tokens"]
    token_score = max(0, min(1, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET))
    valid = sum(1 for r in results if r["answer"])

    # CSV 用合并 token 统计
    output_path = generate_answer_csv_token_stats(
        results,
        plus_stats["prompt_tokens"] + long_stats["prompt_tokens"],
        plus_stats["completion_tokens"] + long_stats["completion_tokens"],
        total_tokens,
    )

    agent.save_cot_trails()

    if vector_indexer:
        vector_indexer.finalize()

    answers = [r["answer"] for r in results if r["answer"]]
    answer_dist = Counter(answers).most_common(15)
    single = [a for a in answers if len(a) == 1]
    single_dist = {c: single.count(c) for c in "ABCD"}

    long_q = sum(1 for t in agent.cot_trails if t.get("strategy") == "qwen_long")
    plus_q = len(agent.cot_trails) - long_q

    print(f"\n📊 V16 评测摘要:")
    print(f"  有效答案: {valid}/{len(questions)}")
    print(f"  qwen-plus 题: {plus_q} | qwen-long 题: {long_q}")
    print(f"  qwen-plus Token: {plus_stats['total_tokens']:,} ({plus_stats['call_count']}次)")
    print(f"  qwen-long Token: {long_stats['total_tokens']:,} ({long_stats['call_count']}次)")
    print(f"  合计 Token: {total_tokens:,}")
    print(f"  TokenScore: {token_score:.4f}")
    print(f"\n  答案分布: {answer_dist}")
    print(f"  单选分布: A={single_dist.get('A',0)} B={single_dist.get('B',0)} "
          f"C={single_dist.get('C',0)} D={single_dist.get('D',0)}")

    print(f"\n  📊 V13参照: Score=39.12 | TokenScore=0.630")
    print(f"  ✅ 结果: {output_path}")


if __name__ == "__main__":
    run_a_board()
