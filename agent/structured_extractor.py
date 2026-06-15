"""结构化信息抽取模块

从原始文档文本中抽取结构化字段：
- 数字型：金额、比例、日期、数量
- 条款型：条款编号+内容摘要
- 表格型：表格数据保留
- 关系型：实体间关系
"""
import os
import re
import json
from agent.config import PROCESSED_DIR


# ============ 数字提取 ============

def extract_numbers(text: str) -> list:
    """提取文档中所有关键数字"""
    numbers = []
    # 金额：万元、亿元、元
    patterns = [
        (r'([\d,.]+)\s*[万亿]*元', 'amount'),
        (r'([\d.]+)%', 'percentage'),
        (r'([\d,.]+)\s*万元', 'amount_wan'),
        (r'([\d,.]+)\s*亿元', 'amount_yi'),
        (r'第([一二三四五六七八九十百千万零〇\d]+)条', 'clause_num'),
        (r'(\d{4})年', 'year'),
        (r'(\d+)月(\d+)日', 'date'),
        (r'(\d+)日', 'day'),
    ]
    for pattern, ntype in patterns:
        for m in re.finditer(pattern, text):
            numbers.append({"type": ntype, "value": m.group(), "context": text[max(0, m.start()-30):m.end()+30]})
    return numbers


# ============ 条款提取 ============

def extract_clauses(text: str) -> list:
    """提取条款编号及内容摘要"""
    clause_pattern = re.compile(r'(第[一二三四五六七八九十百千万零〇\d]+条)\s*(.*?)(?=第[一二三四五六七八九十百千万零〇\d]+条|$)', re.DOTALL)
    clauses = []
    for m in clause_pattern.finditer(text):
        clause_num = m.group(1)
        content = m.group(2).strip()[:200]  # 摘要截断
        if content:
            clauses.append({"number": clause_num, "summary": content})
    return clauses


# ============ 表格提取 ============

def extract_tables(text: str) -> list:
    """提取表格数据（基于制表符或连续数字排列）"""
    tables = []
    lines = text.split('\n')
    table_lines = []
    in_table = False
    
    for line in lines:
        # 检测表格行：包含多个数字+分隔符
        if re.search(r'[\d,.]+\s*[\t│|]\s*[\d,.]+', line) or \
           re.search(r'[\d,.]+\s{2,}[\d,.]+\s{2,}[\d,.]+', line):
            table_lines.append(line)
            in_table = True
        elif in_table and line.strip() and not re.match(r'^[=\-─│|]+$', line):
            table_lines.append(line)
        else:
            if len(table_lines) >= 3:  # 至少3行才算表格
                tables.append('\n'.join(table_lines))
            table_lines = []
            in_table = False
    
    if len(table_lines) >= 3:
        tables.append('\n'.join(table_lines))
    
    return tables


# ============ 领域特定结构化抽取 ============

def extract_insurance_structured(text: str) -> dict:
    """保险文档结构化抽取"""
    result = {
        "clauses": extract_clauses(text),
        "numbers": extract_numbers(text)[:50],
        "tables": extract_tables(text)[:5],
        "key_terms": [],
    }
    
    # 提取保险术语
    terms = re.findall(r'(身故保险金|生存保险金|养老年金|现金价值|账户价值|基本保额|已交保费|退保费用|等待期|犹豫期|免责条款|保险期间|投保年龄|缴费方式|缴费期间)', text)
    result["key_terms"] = list(set(terms))
    
    return result


def extract_regulatory_structured(text: str) -> dict:
    """监管法规结构化抽取"""
    result = {
        "clauses": extract_clauses(text),
        "numbers": extract_numbers(text)[:50],
        "key_terms": [],
        "mandatory_actions": [],
        "prohibited_actions": [],
    }
    
    # 强制性要求
    mandatory = re.findall(r'(?:应当|必须|不得|严禁)[^。；]+[。；]', text)
    result["mandatory_actions"] = mandatory[:20]
    
    # 禁止性规定
    prohibited = re.findall(r'不得[^。；]+[。；]', text)
    result["prohibited_actions"] = prohibited[:10]
    
    # 法规术语
    terms = re.findall(r'(股东大会|董事会|特别决议|普通决议|审议|表决权|监管|处罚|罚款|责令)', text)
    result["key_terms"] = list(set(terms))
    
    return result


def extract_financial_reports_structured(text: str) -> dict:
    """财务报表结构化抽取"""
    result = {
        "numbers": extract_numbers(text)[:80],
        "tables": extract_tables(text)[:10],
        "key_terms": [],
    }
    
    # 财务术语
    terms = re.findall(r'(营业收入|净利润|归母净利润|经营活动.*现金流|总资产|净资产|每股收益|分红|派息|研发投入|同比|环比)', text)
    result["key_terms"] = list(set(terms))
    
    return result


def extract_contracts_structured(text: str) -> dict:
    """金融合同结构化抽取"""
    result = {
        "clauses": extract_clauses(text),
        "numbers": extract_numbers(text)[:50],
        "key_terms": [],
    }
    
    terms = re.findall(r'(发行规模|利率|期限|担保|评级|违约|交叉违约|加速到期|偿付顺序|受托管理人)', text)
    result["key_terms"] = list(set(terms))
    
    return result


def extract_research_structured(text: str) -> dict:
    """研报结构化抽取"""
    result = {
        "numbers": extract_numbers(text)[:50],
        "tables": extract_tables(text)[:5],
        "key_terms": [],
    }
    
    terms = re.findall(r'(市场规模|增长率|渗透率|市占率|预计|预期|同比增长|环比增长|行业趋势)', text)
    result["key_terms"] = list(set(terms))
    
    return result


# ============ 统一抽取入口 ============

EXTRACTORS = {
    "insurance": extract_insurance_structured,
    "regulatory": extract_regulatory_structured,
    "financial_reports": extract_financial_reports_structured,
    "financial_contracts": extract_contracts_structured,
    "research": extract_research_structured,
}


def extract_structured_info(text: str, domain: str) -> dict:
    """从文本中抽取结构化信息"""
    extractor = EXTRACTORS.get(domain, lambda t: {"numbers": extract_numbers(t)[:30]})
    return extractor(text)


def build_structured_extracts():
    """为所有已处理文档构建结构化抽取"""
    print("构建结构化信息抽取...")
    output_dir = os.path.join(PROCESSED_DIR, "structured_extracts")
    os.makedirs(output_dir, exist_ok=True)
    
    total = 0
    for domain in os.listdir(PROCESSED_DIR):
        domain_dir = os.path.join(PROCESSED_DIR, domain)
        if not os.path.isdir(domain_dir) or domain in ("compressed_memory", "structured_extracts"):
            continue
        for root, dirs, files in os.walk(domain_dir):
            for f in files:
                if not f.endswith('.json') or f == "structured_index.json":
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        doc_data = json.load(fh)
                    full_text = doc_data.get("full_text", "")
                    if not full_text:
                        continue
                    
                    doc_id = os.path.splitext(f)[0]
                    structured = extract_structured_info(full_text, domain)
                    structured["doc_id"] = doc_id
                    structured["domain"] = domain
                    structured["text_length"] = len(full_text)
                    
                    out_path = os.path.join(output_dir, f"{doc_id}.json")
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(structured, fh, ensure_ascii=False, indent=2)
                    total += 1
                except Exception as e:
                    pass
    
    print(f"  完成: {total} 个文档的结构化抽取")


if __name__ == "__main__":
    build_structured_extracts()
