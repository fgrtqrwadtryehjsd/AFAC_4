"""V30: 结构化预提取架构

核心理念: 赛题考察"动态记忆压缩"和"Token成本优化"
当前 V22: 每题 42K token(喂大段原文) → TokenScore=0.149
V30 目标: 每题 15K token(喂结构化提取) → TokenScore≈0.70

结构化提取策略(零比赛token,纯规则):
- financial_reports: 主要会计数据表(3K/文档)
- financial_contracts: 发行概况+关键条款索引(5K/文档)
- insurance: 各产品关键条款清单(3K/文档)
- regulatory: 原文(本来就<50K,无需压缩)
- research: 关键段落(< 50K)
"""
import re
from typing import Optional


# ============ 结构化提取函数 ============

def extract_financial_report_structured(text: str) -> str:
    """从年报提取结构化财务数据 — 压缩比 100:1

    目标: 3K 以内包含所有关键财务指标
    """
    sections = []

    # 1. 主要会计数据表 (必含营收/净利润/现金流/研发)
    pos = text.find('主要会计数据')
    if pos < 0:
        pos = text.find('主要财务指标')
    if pos >= 0:
        section = text[pos:pos+3000]
        sections.append(f"[主要会计数据]\n{section}")

    # 2. 研发投入 (选项常考)
    for kw in ['研发投入', '研发费用']:
        pos = text.find(kw)
        if pos >= 0:
            ctx = text[max(0, pos-50):pos+500]
            # 只要含数字的段落
            if re.search(r'\d+(?:亿|万|%)', ctx):
                sections.append(f"[{kw}]\n{ctx}")
                break

    # 3. 分红/回购 (选项常考)
    for kw in ['利润分配', '现金分红', '股份回购']:
        pos = text.find(kw)
        if pos >= 0:
            ctx = text[max(0, pos-30):pos+600]
            if re.search(r'\d', ctx):
                sections.append(f"[{kw}]\n{ctx}")
                break

    result = '\n\n'.join(sections)
    return result[:5000]  # 严格控制在 5K


def extract_financial_contract_structured(text: str) -> str:
    """从募集说明书提取结构化发行要素 — 压缩比 100:1

    目标: 6K 以内包含发行人、评级、规模、关键条款
    """
    sections = []
    seen = set()

    def _add(label, content):
        k = content[:50]
        if k not in seen and len(content) > 20:
            seen.add(k)
            sections.append(f"[{label}]\n{content}")

    # 1. 发行概况首节(含发行人名/股票代码/评级/规模/日期)
    for kw in ['本次发行的基本情况', '发行概况', '第二节', '一、发行人基本情况']:
        pos = text.find(kw)
        if 0 <= pos < 50000:
            _add('发行概况', text[pos:pos+2000])
            break

    # 2. 信用评级 (AAA/AA+ 等)
    for kw in ['主体信用评级', '主体长期信用评级', '债项评级', '信用评级', 'AAA', 'AA+', 'AA-']:
        pos = text.find(kw)
        if 0 <= pos < 100000:
            _add(f'评级[{kw}]', text[max(0,pos-50):pos+400])
            break

    # 3. 股票代码/简称
    import re
    code_match = re.search(r'股票(?:代码|简称)[：:]\s*(\d{6}|\w+)', text[:50000])
    if code_match:
        s = code_match.start()
        _add('股票代码', text[max(0,s-50):s+200])

    # 4. 关键条款 (转股价/赎回/回售/违约/兑付日)
    key_terms = ['转股价格', '赎回条款', '回售条款', '违约事件', '兑付日',
                 '资产减值', '业绩承诺', '票面利率', '向下修正', '违约利息']
    for term in key_terms:
        pos = text.find(term)
        if pos >= 0:
            _add(f'条款[{term}]', text[pos:pos+700])
        if sum(len(s) for s in sections) > 5000:
            break

    result = '\n\n'.join(sections)
    return result[:7000]


def extract_insurance_structured(text: str) -> str:
    """从保险条款提取结构化关键信息

    提取: 产品名称、保险责任、身故保险金、退保条款、保单贷款/借款
    重点: 带数字的条款 (金额/比例/天数)
    """
    sections = []
    seen = set()

    def _add(label, content):
        k = content[:50]
        if k not in seen and len(content) > 20:
            seen.add(k)
            sections.append(f"[{label}]\n{content}")

    # 产品名称(前 500 字)
    _add('产品信息', text[:500])

    # 关键条款 (含数字的优先)
    key_clauses = [
        '身故保险金', '满期生存保险金',
        '退保', '现金价值',
        '保单贷款', '借款',  # 国寿用"借款"
        '犹豫期', '等待期',
        '保险责任', '责任免除',
        '个人养老金',
        '80%', '100%',  # 直接找比例数字
    ]
    for kw in key_clauses:
        # 找所有出现位置, 跳过目录部分(前 8K)
        # 目录行特征: 后面跟 "..." 或单独数字
        for m in re.finditer(re.escape(kw), text):
            if m.start() < 8000:
                continue  # 跳过目录
            ctx = text[max(0, m.start()-30):m.start()+700]
            _add(kw, ctx)
            break
        if sum(len(s) for s in sections) > 4500:
            break

    return '\n\n'.join(sections)[:5500]


def extract_regulatory_structured(text: str) -> str:
    """监管法规: 全文(本来就 <50K,无需大压缩)"""
    # 只做轻度压缩: 去除大量空行和重复标题
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text[:45000]


def extract_research_structured(text: str) -> str:
    """研报: 关键段落提取

    研报通常有 '【投资要点】' 和 '图表' 数据
    """
    sections = []

    # 投资要点 (通常在前 3K)
    pos = text.find('投资要点')
    if pos < 0:
        pos = text.find('核心观点')
    if pos >= 0:
        sections.append(f"[投资要点]\n{text[pos:pos+2000]}")

    # 关键数据段 (含大量数字的段落)
    # 每 1000 字检查一个窗口,取数字密度高的
    high_density = []
    for i in range(0, min(len(text), 80000), 1000):
        chunk = text[i:i+1000]
        nums = len(re.findall(r'\d+(?:\.\d+)?(?:%|亿|万)', chunk))
        if nums >= 5:
            high_density.append((nums, i, chunk))

    high_density.sort(key=lambda x: -x[0])
    for _, pos, chunk in high_density[:5]:
        sections.append(chunk)

    # 研报结尾的盈利预测表
    pos = text.rfind('盈利预测')
    if pos >= 0:
        sections.append(f"[盈利预测]\n{text[pos:pos+1500]}")

    result = '\n\n'.join(sections)
    return result[:20000]  # 研报关键内容控制在 20K


STRUCTURED_EXTRACTORS = {
    'financial_reports': extract_financial_report_structured,
    'financial_contracts': extract_financial_contract_structured,
    'insurance': extract_insurance_structured,
    'regulatory': extract_regulatory_structured,
    'research': extract_research_structured,
}


# ============ 测试 ============

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from agent.indexer import DocumentIndex
    di = DocumentIndex()
    di.load()

    # 测试财报
    t = di.get_doc_full_text('annual_byd_2025_report')
    result = extract_financial_report_structured(t)
    print(f'财报 BYD 2025: {len(t)//1000}K → {len(result)//1000}K')
    print(result[:500])
    print()

    # 测试合同
    t2 = di.get_doc_full_text('text04')
    result2 = extract_financial_contract_structured(t2)
    print(f'合同 text04: {len(t2)//1000}K → {len(result2)//1000}K')
    print(result2[:500])
    print()

    # 测试保险
    t3 = di.get_doc_full_text('1')
    result3 = extract_insurance_structured(t3)
    print(f'保险 doc1: {len(t3)//1000}K → {len(result3)//1000}K')
    print(result3[:300])
