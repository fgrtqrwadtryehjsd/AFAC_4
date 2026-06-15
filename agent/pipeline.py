"""主流水线 V5 — 压缩记忆驱动 + 全面技术整合

流程：
1. 结构化信息抽取（离线，不消耗 API token）
2. 预压缩 A 榜所需文档为记忆摘要（用 qwen-turbo 快速压缩）
3. 构建检索索引
4. 逐题推理：压缩记忆 + RAG + CoT + 答案后处理
5. 生成结果 CSV + 保存 CoT 轨迹
"""
import os
import json
from agent.config import QUESTIONS_DIR, PROCESSED_DIR, RESULTS_DIR, TOKEN_BUDGET, MODEL_NAME
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.memory_compressor import compress_all_documents, load_compressed_memory, MemoryManager
from agent.structured_extractor import build_structured_extracts
from agent.reasoner import ReasoningAgent
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
    # 按题型排序：tf > mcq > multi
    type_order = {"tf": 0, "mcq": 1, "multi": 2}
    questions.sort(key=lambda q: (type_order.get(q["answer_format"], 3), q["domain"]))
    print(f"加载了 {len(questions)} 道 {split} 榜题目")
    return questions


def run_a_board():
    print("=" * 60)
    print("AFAC2026 赛题四 - A 榜评测 V5（压缩记忆驱动）")
    print(f"推理模型: {MODEL_NAME} | Token 预算: {TOKEN_BUDGET:,}")
    print("技术栈：动态摘要抽取 + Prompt Engineering + 思维链")
    print("       + Agent记忆管理 + 上下文窗口优化 + RAG")
    print("       + 结构化信息抽取 + 答案后处理")
    print("=" * 60)

    # Step 1: 结构化信息抽取（离线，不消耗 API token）
    print("\n📊 结构化信息抽取...")
    build_structured_extracts()

    # Step 2: 加载题目，确定需要的文档
    questions = load_questions("A")
    if not questions:
        print("未找到题目")
        return

    needed_doc_ids = set()
    for q in questions:
        for d in q.get("doc_ids", []):
            needed_doc_ids.add(d)
    print(f"A 榜需要 {len(needed_doc_ids)} 个唯一文档")

    # Step 3: 压缩（qwen-plus，质量优先）
    print(f"\n📦 预压缩文档记忆（仅 {len(needed_doc_ids)} 个文档）...")
    compress_client = QwenClient()
    
    compressed = compress_all_documents(compress_client, target_doc_ids=needed_doc_ids)
    compress_tokens = compress_client.get_token_stats()["total_tokens"]
    print(f"  压缩消耗 token: {compress_tokens:,}")

    if not compressed:
        compressed = load_compressed_memory()
    print(f"  压缩记忆: {len(compressed)} 个文档")

    # Step 4: 构建检索索引
    print("\n🔍 构建检索索引（RAG）...")
    doc_index = DocumentIndex()
    doc_index.load()

    # Step 5: 用 qwen-plus 推理（压缩 token 不计入推理模型）
    qwen = QwenClient()  # 用 qwen-plus 推理
    memory_manager = MemoryManager(qwen, compressed)

    # Step 6: 推理
    remaining_budget = TOKEN_BUDGET - compress_tokens
    print(f"\n🧠 开始推理（{MODEL_NAME}）... (剩余预算: {remaining_budget:,} tokens)")
    print("=" * 60)

    agent = ReasoningAgent(qwen, doc_index, memory_manager, token_budget=remaining_budget)

    results = []
    for i, q in enumerate(questions):
        # 全局 token 检查（压缩+推理）
        total_used = compress_tokens + qwen.get_token_stats()["total_tokens"]
        if total_used > TOKEN_BUDGET * 0.95:
            print(f"\n⚠️ Token 接近上限 ({total_used:,})")
            for rq in questions[i:]:
                results.append({"qid": rq["qid"], "answer": ""})
            break

        print(f"[{i+1}/{len(questions)}]", end="")
        result = agent.answer_question(q)
        answer = result["answer"] or extract_answer_from_response(result["raw_response"], q["answer_format"])
        results.append({"qid": q["qid"], "answer": answer})

    # Step 7: 生成结果
    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Token 结算
    total_tokens = compress_tokens + qwen.get_token_stats()["total_tokens"]
    token_score = max(0, min(1, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET))
    valid = sum(1 for r in results if r["answer"])

    output_path = generate_answer_csv_token_stats(
        results, 
        compress_client.get_token_stats()["prompt_tokens"] + qwen.get_token_stats()["prompt_tokens"],
        compress_client.get_token_stats()["completion_tokens"] + qwen.get_token_stats()["completion_tokens"],
        total_tokens
    )

    # 保存 CoT 推理轨迹
    agent.save_cot_trails()

    print(f"\n📊 评测摘要:")
    print(f"  有效答案: {valid}/{len(questions)}")
    print(f"  压缩 Token: {compress_tokens:,}")
    print(f"  推理 Token: {qwen.get_token_stats()['total_tokens']:,} ({MODEL_NAME})")
    print(f"  总 Token: {total_tokens:,}")
    print(f"  TokenScore: {token_score:.4f}")
    print(f"  ✅ 结果: {output_path}")


if __name__ == "__main__":
    run_a_board()
