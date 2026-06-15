"""条款精准定位模块

针对监管法规/合同等结构化文档，通过条款编号精准定位相关条款。
例如题目提到"第47条"，直接定位到该条款文本。
"""
import re


# 匹配中文条款编号
CLAUSE_NUM_PATTERN = re.compile(
    r'第[一二三四五六七八九十百千万零〇\d]+[条章节款项]',
    re.UNICODE
)

# 匹配题目中引用的条款编号
QUESTION_CLAUSE_PATTERN = re.compile(
    r'第[一二三四五六七八九十百千万零〇\d]+条',
    re.UNICODE
)

# 中文数字转阿拉伯数字
CN_NUM_MAP = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '百': 100, '千': 1000, '万': 10000,
}

def cn_to_num(cn: str) -> int:
    """中文数字转阿拉伯数字，如 '四十七' -> 47"""
    if cn.isdigit():
        return int(cn)
    
    result = 0
    current = 0
    for char in cn:
        if char in CN_NUM_MAP:
            val = CN_NUM_MAP[char]
            if val >= 10:
                if current == 0:
                    current = 1
                result += current * val
                current = 0
            else:
                current = current * 10 + val if current >= 10 else current * 10 + val
        else:
            # 未知字符，跳过
            pass
    result += current
    return result


def extract_clause_refs(question: str, options: dict) -> list:
    """
    从题目和选项中提取引用的条款编号
    
    Returns:
        [{"doc_ref": str, "clause_num": int, "clause_text": str}]
        例如 [{"doc_ref": "章程指引", "clause_num": 47, "clause_text": "第四十七条"}]
    """
    refs = []
    seen = set()
    
    # 搜索题干
    for m in QUESTION_CLAUSE_PATTERN.finditer(question):
        text = m.group()
        if text not in seen:
            seen.add(text)
            refs.append({"clause_text": text, "clause_num": _parse_clause_num(text)})
    
    # 搜索选项
    for opt_text in options.values():
        for m in QUESTION_CLAUSE_PATTERN.finditer(opt_text):
            text = m.group()
            if text not in seen:
                seen.add(text)
                refs.append({"clause_text": text, "clause_num": _parse_clause_num(text)})
    
    # 也搜索 "第X章"、"第X节"
    for m in CLAUSE_NUM_PATTERN.finditer(question):
        text = m.group()
        if text not in seen and '条' in text:
            seen.add(text)
            refs.append({"clause_text": text, "clause_num": _parse_clause_num(text)})
    
    return refs


def _parse_clause_num(clause_text: str) -> int:
    """解析条款文本中的编号，如 '第四十七条' -> 47"""
    # 提取中文数字部分
    m = re.match(r'第([一二三四五六七八九十百千万零〇\d]+)条', clause_text)
    if not m:
        return -1
    num_str = m.group(1)
    return cn_to_num(num_str)


def locate_clauses_in_doc(doc_text: str, clause_refs: list) -> list:
    """
    在文档中精准定位引用的条款
    
    Returns:
        [{"clause_text": str, "content": str}] - 条款原文及内容
    """
    if not clause_refs or not doc_text:
        return []
    
    results = []
    for ref in clause_refs:
        clause_text = ref["clause_text"]
        # 在文档中搜索该条款的位置
        pos = doc_text.find(clause_text)
        if pos >= 0:
            # 提取该条款内容（到下一个条款或最多2000字）
            end = len(doc_text)
            next_clause = CLAUSE_NUM_PATTERN.search(doc_text, pos + len(clause_text))
            if next_clause:
                end = next_clause.start()
            content = doc_text[pos:min(end, pos + 2000)].strip()
            results.append({
                "clause_text": clause_text,
                "content": content,
            })
    
    return results


def enrich_evidence_with_clauses(get_doc_full_text_fn, doc_ids: list, clause_refs: list) -> list:
    """
    用条款精准定位补充证据
    
    Args:
        get_doc_full_text_fn: callable(doc_id) -> full_text
    """
    extra_chunks = []
    if not clause_refs or not doc_ids:
        return extra_chunks
    
    for doc_id in doc_ids:
        doc_text = get_doc_full_text_fn(doc_id)
        if not doc_text:
            continue
        
        located = locate_clauses_in_doc(doc_text, clause_refs)
        for loc in located:
            extra_chunks.append({
                "doc_id": doc_id,
                "chunk_id": -1,
                "text": loc["content"],
                "type": "clause_precise",
            })
    
    return extra_chunks
