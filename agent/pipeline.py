"""主流水线 V2 - A榜评测

核心改进：
- 结构化索引 + BM25 + Qwen Rerank
- 领域专用 Prompt
- Token 预算管控（5M预算分级分配）
"""
import os
import json
import time
from agent.config import (
    QUESTIONS_DIR, PROCESSED_DIR, RESULTS_DIR, TOKEN_BUDGET, MODEL_NAME
)
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner import ReasoningAgent
from agent.postprocessor import generate_answer_csv, validate_answer, extract_answer_from_response, generate_answer_csv_token_stats


def load_questions(split: str = "A") -> list:
    """加载题目"""
    questions = []
    questions_dir = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    
    if not os.path.exists(questions_dir):
        print(f"题目目录不存在: {questions_dir}")
        return questions
    
    for filename in sorted(os.listdir(questions_dir)):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(questions_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            domain_questions = json.load(f)
            questions.extend(domain_questions)
    
    # 按领域+题型排序：简单题优先（tf > mcq > multi）
    type_order = {"tf": 0, "mcq": 1, "multi": 2}
    questions.sort(key=lambda q: (type_order.get(q["answer_format"], 3), q["domain"]))
    
    print(f"加载了 {len(questions)} 道 {split} 榜题目")
    return questions


def run_a_board():
    """运行 A 榜评测流水线 V2"""
    print("=" * 60)
    print("AFAC2026 赛题四 - A 榜评测流水线 V2")
    print(f"模型: {MODEL_NAME} | Token 预算: {TOKEN_BUDGET:,}")
    print("=" * 60)

    # Step 1: 初始化 Qwen 客户端
    qwen_client = QwenClient()
    
    # Step 2: 构建索引（含 Qwen Rerank 能力）
    print("\n构建文档索引...")
    doc_index = DocumentIndex(qwen_client=qwen_client)
    doc_index.load()
    
    # Step 3: 加载题目
    questions = load_questions("A")
    if not questions:
        print("未找到题目，退出")
        return
    
    # Step 4: Token 预算分配
    # 保留 20% 给 rerank，80% 给推理
    rerank_budget = int(TOKEN_BUDGET * 0.2)
    reasoning_budget = TOKEN_BUDGET - rerank_budget
    per_question_budget = reasoning_budget // len(questions)
    print(f"\nToken 预算分配: Rerank {rerank_budget:,} + 推理 {reasoning_budget:,}")
    print(f"每题推理预算: ~{per_question_budget:,} tokens")
    
    # Step 5: 初始化推理引擎
    agent = ReasoningAgent(qwen_client, doc_index, token_budget=reasoning_budget)
    
    # Step 6: 逐题推理
    print(f"\n开始推理 {len(questions)} 道题目...")
    print("=" * 60)
    
    results = []
    domain_stats = {}  # 按领域统计
    
    for i, question in enumerate(questions):
        qid = question["qid"]
        domain = question["domain"]
        answer_format = question["answer_format"]
        
        # 检查 token 预算
        stats = qwen_client.get_token_stats()
        if stats["total_tokens"] > TOKEN_BUDGET * 0.95:
            print(f"\n⚠️  Token 预算接近上限 ({stats['total_tokens']:,})，停止推理")
            # 未答题目用空答案
            for remaining_q in questions[i:]:
                results.append({
                    "qid": remaining_q["qid"],
                    "answer": "",
                    "raw_response": "",
                    "evidence_chunks": [],
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })
            break
        
        print(f"[{i+1}/{len(questions)}] {qid} ({domain}/{answer_format})", end=" ")
        
        try:
            result = agent.answer_question(question)
            
            # 答案校验
            answer = result["answer"]
            if not answer:
                answer = extract_answer_from_response(result["raw_response"], answer_format)
            result["answer"] = answer
            
            # 记录 token 消耗
            current_stats = qwen_client.get_token_stats()
            result["tokens"] = {
                "prompt_tokens": current_stats["prompt_tokens"],
                "completion_tokens": current_stats["completion_tokens"],
                "total_tokens": current_stats["total_tokens"],
            }
            
            results.append(result)
            
            # 领域统计
            if domain not in domain_stats:
                domain_stats[domain] = {"count": 0, "answered": 0}
            domain_stats[domain]["count"] += 1
            if answer:
                domain_stats[domain]["answered"] += 1
                
        except Exception as e:
            print(f"✗ 失败: {e}")
            results.append({
                "qid": qid,
                "answer": "",
                "raw_response": str(e),
                "evidence_chunks": [],
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
            if domain not in domain_stats:
                domain_stats[domain] = {"count": 0, "answered": 0}
            domain_stats[domain]["count"] += 1

    # Step 7: 生成结果
    print("\n" + "=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 计算 token 统计：用 qwen_client 的累计统计
    stats = qwen_client.get_token_stats()
    total_prompt = stats["prompt_tokens"]
    total_completion = stats["completion_tokens"]
    total_all = stats["total_tokens"]
    
    # 为每个题目分配 token（按比例分配，基于答案复杂度）
    answer_rows = []
    for r in results:
        answer_rows.append({
            "qid": r["qid"],
            "answer": r["answer"],
        })
    
    output_path = generate_answer_csv_token_stats(
        answer_rows, total_prompt, total_completion, total_all
    )
    
    # 统计摘要
    stats = qwen_client.get_token_stats()
    valid_answers = sum(1 for r in results if r["answer"])
    
    print(f"\n📊 评测摘要:")
    print(f"  有效答案: {valid_answers}/{len(questions)}")
    print(f"  总 Token: {stats['total_tokens']:,}")
    print(f"  API 调用: {stats['call_count']} 次")
    
    token_score = max(0, min(1, (TOKEN_BUDGET - stats["total_tokens"]) / TOKEN_BUDGET))
    print(f"  TokenScore: {token_score:.4f}")
    
    print(f"\n  领域统计:")
    for domain, s in domain_stats.items():
        print(f"    {domain}: {s['answered']}/{s['count']} 有效答案")
    
    print(f"\n✅ 完成！结果文件: {output_path}")
    return output_path


if __name__ == "__main__":
    run_a_board()
