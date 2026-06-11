"""结构化分块器

按条款/章节/表格等语义边界分块，保持语义完整性。
不再使用固定窗口切分。
"""
import os
import json
import re
from agent.config import PROCESSED_DIR


# 条款正则：匹配 "第X条"、"第X章"、"第X节" 等
CLAUSE_PATTERN = re.compile(
    r'第[一二三四五六七八九十百千万零〇\d]+[条章节款项部篇编]',
    re.UNICODE
)

# 章节标题正则
SECTION_PATTERN = re.compile(
    r'^(第[一二三四五六七八九十百千万零〇\d]+[章节篇编]|[一二三四五六七八九十]+[、.．])\s*\S',
    re.MULTILINE | re.UNICODE
)


def split_by_structure(text: str, domain: str) -> list:
    """
    结构化分块：按条款、章节、表格等语义边界切分
    
    Returns:
        [{"chunk_id": int, "text": str, "type": str}]
    """
    chunks = []
    chunk_id = 0

    if domain == "regulatory":
        # 监管法规：以条款为分块单位
        chunks = _split_regulatory(text, chunk_id)
    elif domain == "insurance":
        # 保险条款：以条款+表格为分块单位
        chunks = _split_insurance(text, chunk_id)
    elif domain == "financial_reports":
        # 财务报表：以章节+表格为分块单位
        chunks = _split_financial_reports(text, chunk_id)
    elif domain == "financial_contracts":
        # 金融合同：以条款为分块单位
        chunks = _split_contracts(text, chunk_id)
    elif domain == "research":
        # 研报：以章节为分块单位
        chunks = _split_research(text, chunk_id)
    else:
        # 通用：固定窗口
        chunks = _split_fixed(text, chunk_id, chunk_size=2000)

    return chunks


def _split_regulatory(text: str, start_id: int) -> list:
    """监管法规分块：以条款为最小单位，每块包含1-3条"""
    # 找到所有条款位置
    positions = []
    for m in CLAUSE_PATTERN.finditer(text):
        positions.append(m.start())

    if not positions:
        return _split_fixed(text, start_id, chunk_size=3000)

    chunks = []
    chunk_id = start_id
    # 每块包含连续的条款，最多 3000 字
    i = 0
    while i < len(positions):
        start = positions[i]
        end_char = len(text)
        
        # 找下一个块的起始位置
        char_count = 0
        j = i + 1
        while j < len(positions):
            segment_len = positions[j] - start
            if segment_len > 4000:
                break
            char_count = segment_len
            j += 1
        
        if j < len(positions):
            end_char = positions[j]
        
        chunk_text = text[start:end_char].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "type": "clause",
            })
            chunk_id += 1
        i = j if j > i + 1 else i + 1

    return chunks


def _split_insurance(text: str, start_id: int) -> list:
    """保险条款分块：条款+表格"""
    chunks = []
    chunk_id = start_id
    
    # 按条款分割
    positions = []
    for m in CLAUSE_PATTERN.finditer(text):
        positions.append(m.start())
    
    if not positions:
        # 没有条款标记，按固定窗口
        return _split_fixed(text, start_id, chunk_size=2000)
    
    # 每块1-2个条款
    i = 0
    while i < len(positions):
        start = positions[i]
        end_pos = positions[i + 1] if i + 1 < len(positions) else len(text)
        # 如果单个条款太短，合并下一个
        while i + 1 < len(positions) and (positions[i + 1] - start) < 500:
            i += 1
            end_pos = positions[i + 1] if i + 1 < len(positions) else len(text)
        
        chunk_text = text[start:end_pos].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "type": "clause",
            })
            chunk_id += 1
        i += 1
    
    # 如果还有未匹配的大段文本（如保险产品说明）
    if not chunks:
        chunks = _split_fixed(text, start_id, chunk_size=2000)
    
    return chunks


def _split_financial_reports(text: str, start_id: int) -> list:
    """财务报表分块：按章节+表格"""
    # 章节分割
    positions = []
    for m in SECTION_PATTERN.finditer(text):
        positions.append(m.start())
    
    if not positions:
        return _split_fixed(text, start_id, chunk_size=3000)
    
    chunks = []
    chunk_id = start_id
    for i, pos in enumerate(positions):
        end_pos = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk_text = text[pos:end_pos].strip()
        
        # 如果章节太长，进一步切分
        if len(chunk_text) > 6000:
            sub_chunks = _split_fixed(chunk_text, chunk_id, chunk_size=3000)
            chunks.extend(sub_chunks)
            chunk_id += len(sub_chunks)
        elif chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "type": "section",
            })
            chunk_id += 1
    
    # 确保关键财务数据表被包含（如果前面分块遗漏）
    # 搜索包含"万元"、"亿元"、"比例"的段落
    table_pattern = re.compile(r'(?:万元|亿元|人民币|比例|增长率|同比|环比|占.*比)', re.UNICODE)
    for i, chunk in enumerate(chunks):
        if table_pattern.search(chunk["text"]):
            chunk["type"] = "financial_data"
    
    return chunks


def _split_contracts(text: str, start_id: int) -> list:
    """金融合同分块：按条款"""
    positions = []
    for m in CLAUSE_PATTERN.finditer(text):
        positions.append(m.start())
    
    if not positions:
        return _split_fixed(text, start_id, chunk_size=3000)
    
    chunks = []
    chunk_id = start_id
    i = 0
    while i < len(positions):
        start = positions[i]
        # 每块1-2个条款
        end_idx = min(i + 2, len(positions))
        end_pos = positions[end_idx] if end_idx < len(positions) else len(text)
        chunk_text = text[start:end_pos].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "type": "clause",
            })
            chunk_id += 1
        i = end_idx
    
    return chunks


def _split_research(text: str, start_id: int) -> list:
    """研报分块：按章节"""
    positions = []
    for m in SECTION_PATTERN.finditer(text):
        positions.append(m.start())
    
    if not positions:
        return _split_fixed(text, start_id, chunk_size=3000)
    
    chunks = []
    chunk_id = start_id
    for i, pos in enumerate(positions):
        end_pos = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk_text = text[pos:end_pos].strip()
        if len(chunk_text) > 8000:
            sub = _split_fixed(chunk_text, chunk_id, chunk_size=3000)
            chunks.extend(sub)
            chunk_id += len(sub)
        elif chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "type": "section",
            })
            chunk_id += 1
    
    return chunks


def _split_fixed(text: str, start_id: int, chunk_size: int = 2000) -> list:
    """固定窗口分块（兜底策略）"""
    chunks = []
    chunk_id = start_id
    overlap = min(200, chunk_size // 10)
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "type": "fixed",
            })
            chunk_id += 1
    return chunks


def rebuild_structured_index():
    """用结构化分块重建索引"""
    print("重建结构化索引...")
    index_data = {
        "documents": {},    # doc_id -> full_text
        "doc_meta": {},     # doc_id -> metadata
        "chunks": [],       # [{doc_id, chunk_id, text, type, keywords}]
    }
    
    total_chunks = 0
    for domain in os.listdir(PROCESSED_DIR):
        domain_dir = os.path.join(PROCESSED_DIR, domain)
        if not os.path.isdir(domain_dir):
            continue
        
        for root, dirs, files in os.walk(domain_dir):
            for f in files:
                if not f.endswith('.json'):
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        doc_data = json.load(fh)
                except:
                    continue
                
                doc_id = os.path.splitext(f)[0]
                full_text = doc_data.get("full_text", "")
                if not full_text:
                    continue
                
                index_data["documents"][doc_id] = full_text
                index_data["doc_meta"][doc_id] = {
                    "domain": domain,
                    "source": doc_data.get("source", ""),
                    "total_pages": doc_data.get("total_pages", 0),
                }
                
                # 结构化分块
                structured_chunks = split_by_structure(full_text, domain)
                for chunk in structured_chunks:
                    index_data["chunks"].append({
                        "doc_id": doc_id,
                        "chunk_id": total_chunks + chunk["chunk_id"],
                        "text": chunk["text"],
                        "type": chunk["type"],
                    })
                
                total_chunks += len(structured_chunks)
    
    # 保存索引
    output_path = os.path.join(PROCESSED_DIR, "structured_index.json")
    # 只保存 chunks 和 meta，不保存全文（太大）
    save_data = {
        "doc_meta": index_data["doc_meta"],
        "chunks": index_data["chunks"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False)
    
    print(f"  索引已保存: {len(index_data['documents'])} 个文档, {len(index_data['chunks'])} 个块")
    return index_data


if __name__ == "__main__":
    rebuild_structured_index()
