"""V6 流水线 — Query-Focused Evidence Extraction (QFEE)

核心创新（受 FinCARDS + Acon 论文启发）：
- 不做全局文档压缩（压缩丢信息 = 准确率低）
- 每道题单独提取问题相关的证据段落（QFEE）
- BM25 + 关键词双路召回 → RRF融合 → CoT推理
- 省下压缩 token 用于更精确的推理

Token 预算分配：
- 每题推理：~10K tokens (含 Self-Critique ~2K)
- 总计 100 题：~1.2M（远低于 5M 预算）
"""
import os
import json
from agent.config import QUESTIONS_DIR, PROCESSED_DIR, RESULTS_DIR, TOKEN_BUDGET, MODEL_NAME
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v6 import ReasoningAgentV6
from agent.postprocessor import extract_answer_from_response, generate_answer_csv_token_stats


def load_questions(split: str = "A") -> list:
    questions = []
    questions_dir = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    if not os.path.exists(questions_dir):
        return questions
    for filename in sorted(os.listdir(questions_dir)):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(questions_dir, filename), "r", encoding="utf-8") as f:
            questions.extend(json.load(f))
    return questions


def run_a_board():
    """A 榜评测 V6"""
    print("=" * 60)
    print("AFAC2026 赛题四 - A 榜评测 V6 (QFEE)")
    print(f"推理模型: {MODEL_NAME} | Token 预算: {TOKEN_BUDGET:,}")
    print("技术栈：Query-Focused Evidence Extraction")
    print("       关键词+BM25双路召回 → RRF融合 → CoT推理")
    print("=" * 60)

    # Step 1: 加载题目
    questions = load_questions("A")
    print(f"加载了 {len(questions)} 道 A 榜题目")

    # Step 2: 构建检索索引（无压缩！）
    print("\n🔍 构建检索索引（跳过全局压缩）...")
    doc_index = DocumentIndex()
    doc_index.load()

    # Step 3: 推理
    qwen = QwenClient()
    print(f"\n🧠 开始推理（{MODEL_NAME}）... (预算: {TOKEN_BUDGET:,} tokens)")
    print("=" * 60)

    agent = ReasoningAgentV6(qwen, doc_index, token_budget=TOKEN_BUDGET)

    results = []
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print(f"\n⚠️ Token 接近上限 ({stats['total_tokens']:,})")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break

        print(f"[{i+1}/{len(questions)}]", end=" ")
        result = agent.answer_question(q)
        answer = result["answer"] or extract_answer_from_response(
            result["raw_response"], q["answer_format"])
        results.append({"qid": q["qid"], "answer": answer})
        # 打印答案
        fmt = q.get("answer_format", "")
        domain = q.get("domain", "")
        print(f" {q['qid']} ({domain}/{fmt}) → {answer}")

    # Step 4: 生成结果
    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_tokens = qwen.get_token_stats()["total_tokens"]
    token_score = max(0, min(1, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET))
    valid = sum(1 for r in results if r["answer"])

    output_path = generate_answer_csv_token_stats(
        results, 
        qwen.get_token_stats()["prompt_tokens"],
        qwen.get_token_stats()["completion_tokens"],
        total_tokens
    )

    agent.save_cot_trails()

    print(f"\n📊 评测摘要:")
    print(f"  有效答案: {valid}/{len(questions)}")
    print(f"  推理 Token: {total_tokens:,} ({MODEL_NAME})")
    print(f"  TokenScore: {token_score:.4f}")
    print(f"  ✅ 结果: {output_path}")


if __name__ == "__main__":
    run_a_board()
