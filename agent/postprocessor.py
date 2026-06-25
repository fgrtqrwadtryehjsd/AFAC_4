"""答案后处理与结果生成

处理推理结果，格式校验，生成 answer.csv。
"""
import os
import re
import pandas as pd
from agent.config import RESULTS_DIR


def validate_answer(answer: str, answer_format: str) -> str:
    """
    校验并标准化答案
    
    规则：
    - 单选题：一个大写字母 (A/B/C/D)
    - 判断题：一个大写字母 (A/B)
    - 多选题：多个大写字母按字母序排列，无分隔符 (如 ABC)
    - 非法答案返回空字符串
    """
    if not answer:
        return ""
    
    # 提取所有大写字母
    letters = [c for c in answer.upper() if c in "ABCD"]
    
    if not letters:
        return ""
    
    if answer_format in ("mcq", "tf"):
        # 单选/判断：取首个有效字母
        return letters[0]
    elif answer_format == "multi":
        # 多选：去重排序
        unique = sorted(set(letters))
        return "".join(unique)
    
    return letters[0]


def extract_answer_from_response(response: str, answer_format: str) -> str:
    """
    从模型原始输出中更鲁棒地提取答案

    ⚠️ 历史教训 (2026-06-22): 之前修过一个看似"明显的 bug"(策略 4 .upper() 会把 doc_id
    里的小写英文当成 ABCD), 修了之后 V13 39.12 跌到 25.83. 原因是这个 buggy 行为实际是
    "正确率守门员" — multi 题模型 raw 含明确"最终答案 ABC"时反而引导出过选错误.
    因此**保持 V13 原 extract 逻辑不变**, 别想着"修这个明显 bug".
    """
    # 策略1：寻找"答案"标记
    patterns = [
        r'答案[：:]\s*([A-D]+)',
        r'最终答案[：:]\s*([A-D]+)',
        r'选择[：:]\s*([A-D]+)',
        r'正确答案[：:]\s*([A-D]+)',
        r'answer[：:]\s*([A-D]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return validate_answer(match.group(1), answer_format)

    # 策略2：查找独立成行的字母
    for line in response.strip().split("\n"):
        line = line.strip()
        # 纯字母行如 "AC" 或 "A"
        if re.match(r'^[A-D]+$', line):
            return validate_answer(line, answer_format)

    # 策略3：查找最后出现的选项引用
    all_letters = re.findall(r'选项\s*([A-D])|选项([A-D])|选\s*([A-D])', response)
    if all_letters:
        letters = []
        for groups in all_letters:
            for g in groups:
                if g:
                    letters.append(g)
        if letters:
            return validate_answer("".join(letters), answer_format)

    # 策略4：从全文提取所有大写字母（最后手段）
    # ⚠️ 保持 .upper() — 别改! 之前修这个导致退分
    letters = [c for c in response.upper() if c in "ABCD"]
    if letters:
        return validate_answer("".join(letters), answer_format)

    return ""


def generate_answer_csv(results: list, output_path: str = None):
    """
    生成 answer.csv 提交文件
    
    Args:
        results: 推理结果列表，每项包含 qid, answer, tokens
        output_path: 输出路径
    """
    output_path = output_path or os.path.join(RESULTS_DIR, "answer.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 汇总 Token 统计
    total_prompt = sum(r["tokens"]["prompt_tokens"] for r in results)
    total_completion = sum(r["tokens"]["completion_tokens"] for r in results)
    total_tokens = total_prompt + total_completion
    
    rows = []
    # summary 行
    rows.append({
        "qid": "summary",
        "answer": "",
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
    })
    
    # 每道题一行
    for r in results:
        rows.append({
            "qid": r["qid"],
            "answer": r["answer"],
            "prompt_tokens": r["tokens"]["prompt_tokens"],
            "completion_tokens": r["tokens"]["completion_tokens"],
            "total_tokens": r["tokens"]["total_tokens"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    print(f"总 Token 消耗: {total_tokens:,}")
    print(f"题目数: {len(results)}")
    
    return output_path


def generate_answer_csv_token_stats(answer_rows: list, total_prompt: int, total_completion: int, total_all: int, output_path: str = None):
    """
    生成 answer.csv — 使用全局 token 统计
    
    Args:
        answer_rows: [{"qid": str, "answer": str}]
        total_prompt: 总输入 token
        total_completion: 总输出 token
        total_all: 总 token
    """
    output_path = output_path or os.path.join(RESULTS_DIR, "answer.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = [{
        "qid": "summary",
        "answer": "",
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_all,
    }]
    
    # 每题平均分配 token（天池只看 summary 行的总量）
    n = len(answer_rows)
    per_prompt = total_prompt // n if n else 0
    per_completion = total_completion // n if n else 0
    per_total = total_all // n if n else 0
    
    for r in answer_rows:
        rows.append({
            "qid": r["qid"],
            "answer": r["answer"],
            "prompt_tokens": per_prompt,
            "completion_tokens": per_completion,
            "total_tokens": per_total,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    print(f"总 Token 消耗: {total_all:,}")
    print(f"题目数: {n}")
    
    return output_path
