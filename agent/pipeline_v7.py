"""V7 流水线 — 渐进式记忆压缩 Agent

核心技术创新：
1. 基于规则的结构化Card提取（零Token，预处理阶段）
2. 渐进式记忆：文档首次被访问时构建Card，后续复用
3. Card引导检索：条款精准定位 + 关键词精准匹配
4. 自适应证据量 + 跨题目学习
"""
import os
import json
from agent.config import QUESTIONS_DIR, PROCESSED_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v7 import ReasoningAgentV7
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
    print("=" * 60)
    print("AFAC2026 赛题四 - A 榜评测 V7")
    print("渐进式记忆压缩 Agent (Progressive Memory Compression)")
    print("创新: 规则Card提取(0 Token) + Card引导检索 + 条款精准定位")
    print(f"模型: qwen-plus | Token 预算: {TOKEN_BUDGET:,}")
    print("=" * 60)

    questions = load_questions("A")
    print(f"加载了 {len(questions)} 道 A 榜题目")

    print("\n🔍 构建检索索引...")
    doc_index = DocumentIndex()
    doc_index.load()

    qwen = QwenClient()
    agent = ReasoningAgentV7(qwen, doc_index, token_budget=TOKEN_BUDGET)

    print(f"\n🧠 开始推理 (渐进式记忆压缩)...")
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
        answer = result["answer"] or extract_answer_from_response(
            result["raw_response"], q["answer_format"])
        results.append({"qid": q["qid"], "answer": answer})

        fmt = q.get("answer_format", "")
        domain = q.get("domain", "")
        mem = agent.memory
        clause_refs = agent.cot_trails[-1].get("clause_refs_found", 0)
        print(f" {q['qid']} ({domain}/{fmt}) → {answer} "
              f"[条款:{clause_refs} Card复用:{sum(mem.doc_access_count.values())}]")

    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total_tokens = stats["total_tokens"]
    token_score = max(0, min(1, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET))
    valid = sum(1 for r in results if r["answer"])

    output_path = generate_answer_csv_token_stats(
        results, stats["prompt_tokens"], stats["completion_tokens"], total_tokens)

    agent.save_cot_trails()

    mem = agent.memory
    print(f"\n📊 评测摘要:")
    print(f"  有效答案: {valid}/{len(questions)}")
    print(f"  总 Token: {total_tokens:,}")
    print(f"  TokenScore: {token_score:.4f}")
    print(f"  Card 构建耗时: {mem.card_build_time:.1f}s (零Token)")
    print(f"  Card 文档数: {len(mem.cards)}")
    print(f"  Card 总访问: {sum(mem.doc_access_count.values())} 次")
    print(f"  ✅ 结果: {output_path}")


if __name__ == "__main__":
    run_a_board()
