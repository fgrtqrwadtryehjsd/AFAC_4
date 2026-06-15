"""V8 流水线 — 精确证据推理 Agent

核心突破：
1. 选项去偏 — 打乱ABCD顺序消除A-bias（51/69选A太异常）
2. 全文证据 — qwen-plus 131K tokens ≈ 180K chars上下文窗口
3. 两阶段推理 — 先扫描后验证
"""
import os
import json
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v8 import ReasoningAgentV8
from agent.postprocessor import generate_answer_csv_token_stats


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
    print("=" * 60)
    print("AFAC2026 赛题四 - A 榜评测 V8")
    print("精确证据推理 Agent (Precise Evidence Reasoning)")
    print("突破: 选项去偏(A-bias) + 全文证据(180K chars) + 精确推理")
    print(f"模型: qwen-plus | Token 预算: {TOKEN_BUDGET:,}")
    print("=" * 60)

    questions = load_questions("A")
    print(f"加载了 {len(questions)} 道 A 榜题目")

    print("\n🔍 构建检索索引...")
    doc_index = DocumentIndex()
    doc_index.load()

    qwen = QwenClient()
    agent = ReasoningAgentV8(qwen, doc_index, token_budget=TOKEN_BUDGET)

    print(f"\n🧠 开始推理 (选项去偏 + 全文证据 + 精确推理)...")
    print("=" * 60)

    results = []
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print(f"\n⚠️ Token 接近上限 ({stats['total_tokens']:,})")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break

        print(f"[{i+1}/{len(questions)}]", end="")
        result = agent.answer_question(q)
        answer = result["answer"]
        results.append({"qid": q["qid"], "answer": answer})

        fmt = q.get("answer_format", "")
        domain = q.get("domain", "")
        ev_chars = agent.cot_trails[-1].get("evidence_chars", 0)
        total_doc = agent.cot_trails[-1].get("total_doc_chars", 0)
        print(f" {q['qid']} ({domain}/{fmt}) → {answer} "
              f"[证据:{ev_chars//1000}K/{total_doc//1000}K]")

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total_tokens = stats["total_tokens"]
    token_score = max(0, min(1, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET))
    valid = sum(1 for r in results if r["answer"])

    output_path = generate_answer_csv_token_stats(
        results, stats["prompt_tokens"], stats["completion_tokens"], total_tokens)

    agent.save_cot_trails()

    print(f"\n📊 评测摘要:")
    print(f"  有效答案: {valid}/{len(questions)}")
    print(f"  总 Token: {total_tokens:,}")
    print(f"  TokenScore: {token_score:.4f}")
    print(f"  Card 构建: {agent.memory.card_build_time:.1f}s (零Token)")
    print(f"  ✅ 结果: {output_path}")


if __name__ == "__main__":
    run_a_board()
