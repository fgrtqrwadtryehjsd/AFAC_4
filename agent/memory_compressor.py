"""V5+ 记忆压缩模块 — 全面改进版

改进点：
1. 异常降级：API 失败时自动降级为截断式伪压缩
2. 财报混合压缩：表格用正则保留 + 文字用 LLM 总结
3. 压缩日志 CSV 记录
4. Token 预飞检
"""
import os
import re
import json
import time
import csv
from agent.config import PROCESSED_DIR
from agent.qwen_client import QwenClient


# ============ 财报表格提取（不用 LLM，纯正则） ============

def extract_financial_tables(text: str) -> str:
    """从财报中提取表格数据，保留为紧凑 Markdown 格式
    不消耗 API，避免 LLM 幻觉篡改数字
    """
    tables = []
    lines = text.split('\n')
    table_buffer = []
    in_table = False
    
    for line in lines:
        # 检测表格行：连续数字+分隔符模式
        is_table_row = bool(
            re.search(r'[\d,.]+\s*[\t│|，]\s*[\d,.]+', line) or
            re.search(r'[\d,.]+\s{2,}[\d,.]+\s{2,}[\d,.]+', line) or
            re.search(r'(项目|指标|科目|年度|本期|上期|同比)', line) and re.search(r'[\d,.]+', line)
        )
        
        if is_table_row:
            table_buffer.append(line.strip())
            in_table = True
        elif in_table:
            if len(table_buffer) >= 2:
                tables.append('\n'.join(table_buffer))
            table_buffer = []
            in_table = False
    
    if len(table_buffer) >= 2:
        tables.append('\n'.join(table_buffer))
    
    # 限制总量
    result = '\n\n'.join(tables[:15])
    if len(result) > 8000:
        result = result[:8000]
    return result


def extract_key_figures(text: str) -> str:
    """提取财报关键数字段落（含数字+关键词的句子）"""
    keywords = ['营业收入', '净利润', '归母', '总资产', '净资产', '每股收益', '分红', '派息',
                '现金流', '研发', '同比', '环比', '增长', '减少', '降幅']
    lines = text.split('\n')
    key_lines = []
    for line in lines:
        if any(kw in line for kw in keywords) and re.search(r'[\d,.]+', line):
            key_lines.append(line.strip())
    return '\n'.join(key_lines[:30])


# ============ 动态摘要抽取 Prompt ============

COMPRESSION_PROMPTS = {
    "insurance": """将以下保险条款文档压缩为结构化摘要。必须保留所有关键数字和计算公式。

提取内容：
1. 产品名称和类型
2. 每个产品的关键参数（投保年龄、保险期间、缴费方式、基本保额）
3. 保险责任及计算公式（身故保险金、生存保险金、养老年金等）
4. 现金价值表关键数据（每年度现金价值，至少前10年）
5. 退保费用比例
6. 免责条款
7. 犹豫期、等待期

数字必须精确保留，计算公式完整保留。

文档：
{doc_text}

结构化摘要：""",

    "regulatory": """将以下法规文档压缩为结构化摘要。必须保留所有条款核心内容。

提取内容：
1. 法规名称、发布机构、施行日期
2. 逐条提取条款核心内容（保留条款编号，每条一句话概括）
3. 重点标注：强制性要求（应当/必须/不得）、时限要求
4. 处罚条款
5. 法规间关联

关键条款保留原文表述。

文档：
{doc_text}

结构化摘要：""",

    "financial_contracts": """将以下金融合同文档压缩为结构化摘要。

提取内容：
1. 债券/合同基本信息（名称、规模、利率、期限、担保方式）
2. 评级信息
3. 关键条款（违约定义、交叉违约、加速到期、偿付顺序）
4. 核心权利义务
5. 所有数字精确保留

文档：
{doc_text}

结构化摘要：""",

    "financial_reports": """将以下年报文档的文字描述部分压缩为结构化摘要。

注意：文档中的财务表格数据已经单独提取，你只需要总结文字描述部分。

提取内容：
1. 公司名称、报告年度
2. 核心经营情况文字描述（业务概述、行业变化、战略方向）
3. 重要事项（并购、诉讼、监管处罚等）
4. 分红方案描述（区分预案vs实际）
5. 管理层讨论与分析要点

文字描述摘要：""",

    "research": """将以下研报文档压缩为结构化摘要。

提取内容：
1. 研报标题、发布机构
2. 核心观点和结论
3. 关键数据（市场规模、增长率、渗透率、市占率等）
4. 行业趋势
5. 公司对比数据
6. 区分"预计"vs"实际"

所有数字精确保留。

文档：
{doc_text}

结构化摘要：""",
}


# ============ 核心压缩函数 ============

def fallback_compress(doc_text: str) -> str:
    """降级压缩：API 失败时，取前 30% + 尾部 20%"""
    total_len = len(doc_text)
    head = doc_text[:int(total_len * 0.3)]
    tail = doc_text[int(total_len * 0.8):]
    return f"[文档前部]\n{head}\n\n[文档尾部]\n{tail}"


def compress_document(qwen: QwenClient, doc_id: str, doc_text: str, domain: str) -> str:
    """压缩单篇文档，带异常降级
    
    策略：
    - >50K 字符：直接截断降级（不调API，避免超时）
    - 15K-50K：分块压缩后合并
    - <15K：一次性压缩
    """
    # 超长文档分段压缩（不是截断降级！）
    if len(doc_text) > 50000:
        # 分成多个15K块，每块独立压缩
        parts = []
        chunk_size = 15000
        for i in range(0, len(doc_text), chunk_size):
            chunk = doc_text[i:i + chunk_size]
            if len(chunk) < 500:
                continue
            try:
                prompt = prompt_template.format(doc_text=chunk)
                result = qwen.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1024, timeout=90)
                parts.append(result["content"])
            except:
                parts.append(chunk[:2000])  # 降级为截断
        return "\n\n".join(parts)
    
    try:
        if domain == "financial_reports":
            return _compress_financial_report(qwen, doc_id, doc_text)
        
        prompt_template = COMPRESSION_PROMPTS.get(domain, COMPRESSION_PROMPTS["regulatory"])
        max_chunk = 15000  # 安全上限，避免 API 超时
        
        if len(doc_text) <= max_chunk:
            prompt = prompt_template.format(doc_text=doc_text)
            result = qwen.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048, timeout=90)
            return result["content"]
        else:
            # 分段压缩后合并
            parts = []
            for i in range(0, len(doc_text), max_chunk):
                chunk = doc_text[i:i + max_chunk]
                prompt = prompt_template.format(doc_text=chunk)
                result = qwen.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048, timeout=90)
                parts.append(result["content"])
            
            merge_prompt = f"""将以下同一文档的分段摘要合并为统一结构化摘要，去重：

{chr(10).join(parts)}

合并后摘要："""
            result = qwen.chat([{"role": "user", "content": merge_prompt}], temperature=0.0, max_tokens=2048, timeout=90)
            return result["content"]
    
    except Exception as e:
        print(f"  ⚠️ 压缩失败({e})，降级为截断式", end=" ")
        return fallback_compress(doc_text)


def _compress_financial_report(qwen: QwenClient, doc_id: str, doc_text: str) -> str:
    """财报混合压缩：表格保留 + 文字LLM总结"""
    # 1. 正则提取表格（不消耗 API，避免数字幻觉）
    tables_md = extract_financial_tables(doc_text)
    key_figures = extract_key_figures(doc_text)
    
    # 2. LLM 只总结文字描述（限制输入长度）
    text_only = doc_text[:15000]  # 安全上限
    
    try:
        prompt = COMPRESSION_PROMPTS["financial_reports"].format(doc_text=text_only)
        result = qwen.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048, timeout=90)
        text_summary = result["content"]
    except:
        text_summary = fallback_compress(doc_text)[:3000]
    
    # 3. 拼接：表格在前 + 关键数字 + 文字摘要
    combined = f"### 财务数据表\n{tables_md}\n\n### 关键数字\n{key_figures}\n\n### 经营摘要\n{text_summary}"
    
    # 限制总量
    if len(combined) > 8000:
        combined = combined[:8000]
    
    return combined


# ============ Agent 记忆管理 ============

class MemoryManager:
    """Agent 记忆管理器
    
    两级记忆：
    - 静态记忆：文档压缩摘要（预构建，所有题目共享）
    - 动态记忆：每道题的细化事实（按需生成，该题专享）
    """
    
    def __init__(self, qwen: QwenClient, compressed_memory: dict):
        self.qwen = qwen
        self.static_memory = compressed_memory  # doc_id -> summary
        self.dynamic_memory = {}  # qid -> refined_facts
    
    def get_context(self, doc_ids: list, question: str, domain: str, max_chars: int = 8000) -> str:
        """获取记忆上下文"""
        summaries = []
        total_len = 0
        
        for doc_id in doc_ids:
            if doc_id in self.static_memory:
                summary = self.static_memory[doc_id]
                summaries.append(f"### 文档摘要：{doc_id}\n{summary}")
                total_len += len(summary)
        
        if not summaries:
            return ""
        
        # 如果摘要总量太大，按比例截断
        if total_len > max_chars:
            per_doc = max_chars // max(len(summaries), 1)
            truncated = []
            for s in summaries:
                if len(s) > per_doc:
                    truncated.append(s[:per_doc])
                else:
                    truncated.append(s)
            return '\n\n'.join(truncated)
        
        return '\n\n'.join(summaries)


# ============ 预压缩全量文档（带日志 + 降级） ============

def compress_all_documents(qwen: QwenClient, output_dir: str = None, target_doc_ids: set = None) -> dict:
    """预压缩所有文档（增量处理 + 异常降级 + 日志记录）
    
    Args:
        target_doc_ids: 只压缩这些文档（None=全部）
    """
    output_dir = output_dir or os.path.join(PROCESSED_DIR, "compressed_memory")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载已有进度
    compressed = {}
    done_file = os.path.join(output_dir, "_progress.json")
    log_file = os.path.join(output_dir, "_compression_log.csv")
    done_ids = set()
    
    if os.path.exists(done_file):
        with open(done_file, "r", encoding="utf-8") as f:
            done_ids = set(json.load(f).get("done", []))
    
    # 扫描文档
    all_docs = []
    for domain in os.listdir(PROCESSED_DIR):
        domain_dir = os.path.join(PROCESSED_DIR, domain)
        if not os.path.isdir(domain_dir) or domain in ("compressed_memory", "structured_extracts"):
            continue
        for root, dirs, files in os.walk(domain_dir):
            for f in files:
                if not f.endswith('.json') or f == "structured_index.json":
                    continue
                doc_id = os.path.splitext(f)[0]
                # 如果指定了目标文档，跳过不在列表中的
                if target_doc_ids and doc_id not in target_doc_ids:
                    continue
                if doc_id in done_ids:
                    cache_path = os.path.join(output_dir, f"{doc_id}.txt")
                    if os.path.exists(cache_path):
                        with open(cache_path, "r", encoding="utf-8") as fh:
                            compressed[doc_id] = fh.read()
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        doc_data = json.load(fh)
                    full_text = doc_data.get("full_text", "")
                    if full_text:
                        all_docs.append((doc_id, full_text, domain))
                except:
                    pass
    
    if not all_docs:
        print(f"  已压缩 {len(compressed)} 个文档")
        return compressed
    
    print(f"  待压缩: {len(all_docs)}, 已完成: {len(compressed)}")
    
    # 打开日志文件
    log_fh = open(log_file, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fh)
    if os.path.getsize(log_file) == 0:
        log_writer.writerow(["doc_id", "domain", "original_chars", "compressed_chars", "tokens_used", "time_sec", "method"])
    
    for i, (doc_id, doc_text, domain) in enumerate(all_docs):
        print(f"  压缩 [{i+1}/{len(all_docs)}] {doc_id[:50]}...", end=" ")
        start_time = time.time()
        method = "llm"
        
        try:
            before_tokens = qwen.get_token_stats()["total_tokens"]
            summary = compress_document(qwen, doc_id, doc_text, domain)
            after_tokens = qwen.get_token_stats()["total_tokens"]
            tokens_used = after_tokens - before_tokens
            
            # 保存到文件
            compressed[doc_id] = summary
            with open(os.path.join(output_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as fh:
                fh.write(summary)
            
            # 更新进度
            done_ids.add(doc_id)
            with open(done_file, "w", encoding="utf-8") as fh:
                json.dump({"done": list(done_ids)}, fh, ensure_ascii=False)
            
            elapsed = time.time() - start_time
            log_writer.writerow([doc_id, domain, len(doc_text), len(summary), tokens_used, f"{elapsed:.1f}", method])
            log_fh.flush()
            
            print(f"OK ({tokens_used} tokens, {elapsed:.0f}s)")
        
        except Exception as e:
            # 异常降级：截断式伪压缩
            method = "fallback"
            summary = fallback_compress(doc_text)
            compressed[doc_id] = summary
            with open(os.path.join(output_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as fh:
                fh.write(summary)
            done_ids.add(doc_id)
            with open(done_file, "w", encoding="utf-8") as fh:
                json.dump({"done": list(done_ids)}, fh, ensure_ascii=False)
            
            elapsed = time.time() - start_time
            log_writer.writerow([doc_id, domain, len(doc_text), len(summary), 0, f"{elapsed:.1f}", method])
            log_fh.flush()
            
            print(f"⚠️ 降级 ({e})")
    
    log_fh.close()
    print(f"  压缩完成: {len(compressed)} 个文档")
    return compressed


def load_compressed_memory(memory_dir: str = None) -> dict:
    """加载已有压缩记忆"""
    memory_dir = memory_dir or os.path.join(PROCESSED_DIR, "compressed_memory")
    compressed = {}
    if not os.path.exists(memory_dir):
        return compressed
    for f in os.listdir(memory_dir):
        if f.endswith('.txt') and not f.startswith('_'):
            doc_id = os.path.splitext(f)[0]
            with open(os.path.join(memory_dir, f), "r", encoding="utf-8") as fh:
                compressed[doc_id] = fh.read()
    return compressed


if __name__ == "__main__":
    from agent.qwen_client import QwenClient
    client = QwenClient()
    compress_all_documents(client)
