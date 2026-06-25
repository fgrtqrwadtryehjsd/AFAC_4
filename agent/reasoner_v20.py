"""V20: 大重构 — 题型 × 域分流 + 关键章节优先 + 防过选硬约束.

## 核心策略

V13 失败根因: BM25/vector 散点检索, **关键数字根本不在 prompt 里**.
实测 (financial_reports 域):
- fin_a_001 V13 prompt 含 803,964 (BYD 2025 营收) = 0 次
- fin_a_001 V13 prompt 含 777,102 (BYD 2024 营收) = 0 次
- 模型在没数据时被迫瞎答 → 准确率 29%

V20 策略:
1. **位置感知证据组装**: 每文档取"前 50K + 关键章节" 而不是 RAG 散点
2. **题型专属 prompt**: tf / mcq / multi 不同
3. **域感知补全**: 财报 → 主要会计数据章节; 合同 → 发行概况+条款; 研报+保险+法规 → 前 50K 全文
4. **保留 V13 后处理**: extract_answer + _post_process 不动 (V13 守门效应不能丢)
"""
import re
from collections import defaultdict
from agent.config import RESULTS_DIR
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.postprocessor import extract_answer_from_response


# ============ 域专属证据提取 ============

def _take_head(text: str, n_chars: int) -> str:
    """取前 N 字符 (保持完整, 不切句)"""
    return text[:n_chars]


def _locate_section(text: str, anchors: list, ctx_chars: int = 12000) -> str:
    """找第一个 anchor 出现的位置, 取该位置后 ctx_chars 字符"""
    for anchor in anchors:
        pos = text.find(anchor)
        if pos >= 0:
            return text[pos: pos + ctx_chars]
    return ""


def extract_evidence_financial_report(text: str, max_chars: int = 45000) -> str:
    """财报: 前 25K (含主要会计数据) + 找补关键章节"""
    head = _take_head(text, 25000)
    # 找补 "主要会计数据" (若不在前 25K 内)
    key_section = _locate_section(text, ["主要会计数据", "主要财务指标"], ctx_chars=10000)
    if key_section and key_section[:50] not in head:
        # 已在 head 里则不重复
        result = head + "\n\n[主要会计数据章节]\n" + key_section
    else:
        result = head
    # 找补现金流量/分红章节
    cash_section = _locate_section(text, ["现金流量净额", "现金分红", "利润分配"], ctx_chars=6000)
    if cash_section and cash_section[:50] not in result:
        result += "\n\n[现金流/分红章节]\n" + cash_section
    return result[:max_chars]


def extract_evidence_financial_contract(text: str, max_chars: int = 45000) -> str:
    """合同: 前 35K (含发行概况) + 找补条款"""
    head = _take_head(text, 35000)
    # 找补转股/赎回/违约条款 (有些合同条款在 40-60K 位置)
    for anchor in ["转股价格的修正", "向下修正", "赎回条款", "回售条款", "违约事件"]:
        sec = _locate_section(text, [anchor], ctx_chars=4000)
        if sec and sec[:50] not in head:
            head += "\n\n[条款]\n" + sec
            if len(head) >= max_chars:
                break
    return head[:max_chars]


def extract_evidence_research(text: str, max_chars: int = 45000) -> str:
    """研报: 前 45K (大部分研报 < 50K, 关键数字在前)"""
    return _take_head(text, max_chars)


def extract_evidence_insurance(text: str, max_chars: int = 45000) -> str:
    """保险: 前 45K (保险条款多在前部)"""
    return _take_head(text, max_chars)


def extract_evidence_regulatory(text: str, max_chars: int = 45000) -> str:
    """法规: 全文 (法规通常 < 50K)"""
    return _take_head(text, max_chars)


DOMAIN_EXTRACTORS = {
    "financial_reports": extract_evidence_financial_report,
    "financial_contracts": extract_evidence_financial_contract,
    "research": extract_evidence_research,
    "insurance": extract_evidence_insurance,
    "regulatory": extract_evidence_regulatory,
}


# ============ Prompts ============

DOMAIN_SYSTEM = {
    "insurance": "你是保险条款分析师. 关键: 身故保险金 ≠ 已交保费 ≠ 现金价值 ≠ 账户价值. 退保金额 = 现金价值 − 退保费用. 计算题须先列出每个产品适用的条款和数值, 再算. 选项的具体数字必须在文档中存在原文支持.",
    "regulatory": "你是金融监管合规专家. 关键: \"应当\"\"必须\"\"不得\"=强制; \"可以\"=授权; \"大额交易报告\"≠\"可疑交易报告\"; 时限 (30 个工作日/6 个月/10 年) 须精确匹配原文. 任何用词差异立刻判错.",
    "financial_contracts": "你是金融合同分析师. 关键: 主体信用评级 ≠ 债项信用评级; 第一份文档与第二份文档必须分别核验, 不能张冠李戴. 数字/评级/期限必须在指定文档中原文出现.",
    "financial_reports": "你是财务报表分析师. 关键: 看主要会计数据表; 同比 (2024 vs 2023) ≠ 环比; 归母净利润 ≠ 净利润; 经营/投资/筹资现金流严格区分; 必须找到具体数字才比较大小, 不能凭印象判断.",
    "research": "你是行业研报分析师. 关键: \"预期/预计\" ≠ \"实际\"; 同比增速 ≠ 环比增速; 行业数据 ≠ 公司数据; 具体百分比/金额/年份必须在原文出现才能选.",
}


COMMON_HEADER = """你的任务: 严格依据下列文档证据回答问题. 数字、年份、百分比、用词的任何差异都视为错.

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}
"""

PROMPT_TF = COMMON_HEADER + """
## 判断流程
1. 把上述陈述拆成可独立核验的子陈述 (按"且"/"同时"/";"/","拆分).
2. 对每个子陈述, 在原文中找精确出现的语句, 比较数字/用词/时限.
3. 任何一个子陈述: 数字不符 / 用词被替换 (如"可疑"换成"大额") / 时限不一致 / 主体错位 → 判 B 错误.
4. 全部子陈述与原文完全一致 → 判 A 正确.

## 输出格式 (必须遵守)
子陈述 1: <拆分>
原文核验: <引用>
判定: ✓/✗

子陈述 2: ...
...

最终答案: A 或 B
"""

PROMPT_MCQ = COMMON_HEADER + """
## 判断流程
1. 对 A B C D 4 个选项**全部分析**, 不能只看 A 就停.
2. 每个选项在原文中找直接对应语句, 比较数字/用词.
3. 选**最直接被原文支持**的选项. 警惕"相似但用词有差"陷阱.

## 输出格式 (必须遵守)
选项 A: <分析>, 原文: <引用>, 判定: ✓/✗
选项 B: ...
选项 C: ...
选项 D: ...

最终答案: A 或 B 或 C 或 D
"""

PROMPT_MULTI = COMMON_HEADER + """
## 判断流程 (多选, 必须谨慎)
1. 评分规则: 完全匹配才得分. 漏选/过选都 0 分. **宁可漏选, 不可过选**.
2. 对 A B C D 4 个选项**全部分析** (不能早停).
3. 选项 X 必须满足: 原文中存在与该选项**关键数字/用词/事实完全一致**的语句, 才能选.
4. 数字微差 / 用词替换 / 时限不一致 → ✗ 不选.
5. 涉及"两份文档均..."的选项, 必须**两份文档都能找到**, 缺一不可.

## 输出格式 (必须遵守)
选项 A: <原文引用>, 判定: ✓/✗
选项 B: <原文引用>, 判定: ✓/✗
选项 C: <原文引用>, 判定: ✓/✗
选项 D: <原文引用>, 判定: ✓/✗

最终答案: <按字母序拼接所有 ✓ 选项, 如 ABC. 若无 ✓ 选项, 输出 A>
"""


def build_evidence(doc_index: DocumentIndex, domain: str, doc_ids: list, n_docs: int, max_evidence: int) -> str:
    """按域提取证据, 控制总长度"""
    extractor = DOMAIN_EXTRACTORS.get(domain, extract_evidence_regulatory)
    # 每 doc 配额
    per_doc = max_evidence // max(1, n_docs)
    evidence = ""
    for did in doc_ids:
        t = doc_index.get_doc_full_text(did) or ""
        if not t: continue
        seg = extractor(t, max_chars=per_doc)
        evidence += f"\n=== 文档 {did} ===\n{seg}\n"
    return evidence


# ============ Reasoner ============

class ReasoningAgentV20:
    """V20: 域 + 题型分流"""

    # 每题最大证据量 (留足 LLM prompt 余地)
    MAX_EVIDENCE = {
        "tf": 80000,    # tf 题难度高, 给足上下文
        "mcq": 90000,
        "multi": 90000,
    }

    def __init__(self, qwen: QwenClient, doc_index: DocumentIndex,
                 vector_indexer=None, token_budget: int = 5_000_000, model: str = "qwen-plus"):
        self.qwen = qwen
        self.doc_index = doc_index
        self.vector = vector_indexer  # 不使用, 兼容接口
        self.token_budget = token_budget
        self.model = model
        self.cot_trails = []

        # 兼容旧入口
        from agent.reasoner_v13 import AgentMemory
        self.memory = AgentMemory(doc_index)
        self.FULL_DOC_THRESHOLD = 50000  # 兼容 pipeline 打印

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        max_ev = self.MAX_EVIDENCE.get(answer_format, 80000)

        # 构造证据
        evidence = build_evidence(self.doc_index, domain, doc_ids, len(doc_ids), max_ev)

        # 选 prompt
        if answer_format == "tf":
            prompt_tpl = PROMPT_TF
        elif answer_format == "mcq":
            prompt_tpl = PROMPT_MCQ
        else:
            prompt_tpl = PROMPT_MULTI

        prompt = prompt_tpl.format(
            evidence=evidence,
            question=q_text,
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
        )

        system = DOMAIN_SYSTEM.get(domain, "")

        try:
            result = self.qwen.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""

        answer = extract_answer_from_response(raw_response, answer_format)
        answer = self._post_process(answer, answer_format)

        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": answer,
            "answer_format": answer_format,
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": total_doc_chars <= max_ev,
            "raw_response": raw_response,
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
        }

    def _post_process(self, answer: str, answer_format: str) -> str:
        """继承 V13 守门行为 — 不要改"""
        if not answer:
            return "A" if answer_format in ("mcq", "tf") else ""
        valid = set("ABCD")
        if answer_format == "mcq":
            for c in answer:
                if c in valid:
                    return c
            return "A"
        elif answer_format == "tf":
            if "A" in answer:
                return "A"
            if "B" in answer:
                return "B"
            return "A"
        elif answer_format == "multi":
            letters = sorted(set(c for c in answer if c in valid))
            if len(letters) > 3:
                letters = letters[:3]
            return "".join(letters) if letters else "A"
        return answer

    def save_cot_trails(self, path=None):
        """兼容 pipeline_v13 接口"""
        import os, json
        path = path or os.path.join(RESULTS_DIR, "eval_results_v20.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 截断 raw_response 防文件过大
        out = []
        for t in self.cot_trails:
            t2 = dict(t)
            if "raw_response" in t2:
                t2["raw_response"] = t2["raw_response"][:1500]
            out.append(t2)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  ✅ COT trails -> {path}")
