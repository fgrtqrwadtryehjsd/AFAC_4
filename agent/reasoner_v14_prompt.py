"""V14: 严格 prompt 改造, 不动检索, 不动 retriever, 不动段顺序.

V13.1 退分教训: 改了检索/段排序 → 模型推理路径变 → 5/5 fc tf 翻 + 多选过选 → 退分

V14 设计原则:
1. 100% 继承 V13 的检索/RRF/证据组装/段顺序
2. 只改 COT_PROMPT 文本
3. 改造点直接对应"V13 离线诊断的 4 类错题模式"

V13 错题模式 (基于 100 题 prompt 人工审视):
- 模式 1: tf 复合陈述+关键词替换 (reg_a_010 大额交易 vs 可疑交易, res_a_006 "除"客户资金杠杆 vs 客户资金杠杆)
  → 模型没逐字核验
- 模式 2: tf 复合陈述+部分子陈述无证据 (reg_a_018 "分类评价"原文 0 次)
  → 模型对无证据子陈述不主动否定
- 模式 3: multi 早停 (ins_a_019 验证 A 就停)
  → 模型未强制核验 4 个选项
- 模式 4: multi 单选孤立 A (V13 33 道 multi 答单选, 其中 12 道纯 A)
  → A 偏置 + 早停

V14 改造:
1. tf 题: 用专用 prompt, 强制"将题干拆成 N 个子陈述, 逐一核验"
2. multi 题: 强制"为每个选项都给原文引用或'无原文支持→❌'"
3. 显式"用词陷阱"列表: 同义词/量词/否定/范围
4. 显式"宁可漏选不可过选"约束 (multi 评分完全匹配, 漏选错少, 过选错多)
"""

from agent.reasoner_v13 import (
    ReasoningAgentV13,
    DOMAIN_SYSTEM_PROMPTS,
    CLAUSE_PATTERN,
    extract_query_keywords,
    keyword_match_score,
    compress_whitespace,
    SELF_CRITIQUE_PROMPT,
)
from agent.postprocessor import extract_answer_from_response


# ============ V14 Prompt Templates ============

# 多选题用 — 强制对 4 个选项都给分析, 明确"无原文→❌"
MULTI_COT_PROMPT = """## 任务
严格依据文档证据,逐选项判断每个选项的正确性. 这是多选题, **任一选项过选/漏选都计 0 分,所以必须谨慎**.

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 验证规则 (必须严格执行)

**规则 1**: 对**每一个选项 (A B C D 全部)**, 必须在证据中搜索原文支持. 不能跳过任何选项.

**规则 2**: 判断标准:
- ✅ 明确支持: 证据中有**与选项表述高度一致**的原文 (允许略有改写, 但**关键词、数字、范围、限定语必须一致**)
- ❌ 无原文支持 / 原文相反 / 原文有关键差异: 选项不选
- 关键陷阱: "应当 vs 可以"、"不得 vs 可"、"全部 vs 部分"、"全体 vs 多数"、"大额交易 vs 可疑交易"、"分类评价 vs 监管评级"、"特别决议 vs 普通决议" 等用词差异

**规则 3**: **宁可漏选, 不可过选**. 如果某选项原文证据**不充分或模糊**, 倾向于 ❌ 不选.

**规则 4**: 多 doc 题目, 注意每个选项所指的具体文档 (如"第一份文档"="第二份文档"). 不能将 doc A 的事实归到 doc B 上.

## 逐选项分析 (4 个选项必须全部分析)

**选项 A**: {option_a}
- 搜索关键词:
- 原文引用 (若有):
- 关键词/数字/限定语比对:
- 判断: ✅ 支持 / ❌ 不支持

**选项 B**: {option_b}
- 搜索关键词:
- 原文引用 (若有):
- 关键词/数字/限定语比对:
- 判断: ✅ 支持 / ❌ 不支持

**选项 C**: {option_c}
- 搜索关键词:
- 原文引用 (若有):
- 关键词/数字/限定语比对:
- 判断: ✅ 支持 / ❌ 不支持

**选项 D**: {option_d}
- 搜索关键词:
- 原文引用 (若有):
- 关键词/数字/限定语比对:
- 判断: ✅ 支持 / ❌ 不支持

最终答案 (按字母序, 只包含 ✅ 的选项, 无分隔符):"""


# 判断题用 — 强制将题干拆成子陈述, 任一错则全错
TF_COT_PROMPT = """## 任务
严格依据文档证据, 判断陈述的正确性. 这是**判断题**, 只要陈述中任何一部分与原文不符, 整体就是错误的.

## 文档证据
{evidence}

## 待判断陈述
{question}

## 选项
A. 正确
B. 错误

## 验证规则 (必须严格执行)

**规则 1**: **将陈述拆成可独立核验的子陈述**. 如:
- 复合陈述 "X 应当 Y, 且 W 应当 Z" → 拆成 "X 应当 Y" 和 "W 应当 Z" 两个
- 含数字/范围/比例的陈述 → 数字必须与原文完全一致

**规则 2**: 对**每一个子陈述**, 在证据中找原文核验:
- ✅ 原文明确支持
- ❌ 原文相反 / 原文有关键差异 (用词、数字、范围)
- ❓ 原文未涉及 (无证据)

**规则 3**: 整体判断:
- 所有子陈述都 ✅ → A 正确
- **任一子陈述 ❌ 或 ❓ → B 错误**
- **❓ 也判错** — 不能因为"原文未提及就当作正确"

**规则 4**: 常见陷阱:
- 同义近似词替换: "大额交易" vs "可疑交易", "分类评价" vs "监管评级", "客户资金杠杆" vs "除客户资金杠杆"
- 量词/范围: "全部" vs "部分", "全体" vs "多数", "10 日内" vs "15 日内"
- 时间: "年度" vs "半年度", "披露" vs "审议"
- 主体: "董事会" vs "股东会", "上市公司" vs "证券公司"

## 子陈述拆解与核验

子陈述 1: <拆解此处>
- 原文核验: <引用或"无证据">
- 判断: ✅ / ❌ / ❓

子陈述 2: <拆解此处>
- 原文核验:
- 判断:

(...如有更多子陈述, 继续列)

## 整体判断
- 若所有子陈述都 ✅ → 选 A
- 若有任一 ❌ 或 ❓ → 选 B

最终答案 (A 或 B):"""


# 单选题用 — 类似多选但只选 1 个
MCQ_COT_PROMPT = """## 任务
严格依据文档证据, 从 4 个选项中选出**唯一正确**的一个.

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 验证规则

**规则 1**: 对 4 个选项**全部**搜原文支持, 不能只看选项 A 就停.

**规则 2**: 选**原文支持最直接、用词最一致**的那个. 警惕"听起来对但用词有差"的陷阱.

**规则 3**: 关键陷阱: 同义近似词 (大额 vs 可疑, 分类 vs 监管), 量词 (全部 vs 部分), 否定 (应当 vs 不得), 主体 (董事会 vs 股东会).

## 逐选项分析

**选项 A**: {option_a}
- 原文引用:
- 判断: 选 / 不选

**选项 B**: {option_b}
- 原文引用:
- 判断:

**选项 C**: {option_c}
- 原文引用:
- 判断:

**选项 D**: {option_d}
- 原文引用:
- 判断:

最终答案 (A/B/C/D 单个字母):"""


class ReasoningAgentV14(ReasoningAgentV13):
    """V14: 仅改 prompt, 其他 100% 继承 V13"""

    ENABLE_SELF_CRITIQUE = False  # 关掉以省 token; 改后 prompt 已经"宁可漏选"

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        card_hints = self.memory.get_card_match_hints(q_text, options, doc_ids)

        total_doc_chars = sum(
            self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)
        is_full_doc = total_doc_chars <= self.FULL_DOC_THRESHOLD

        # 证据收集 — 100% 继承 V13
        if is_full_doc:
            evidence_text = ""
            for doc_id in doc_ids:
                ft = self.doc_index.get_doc_full_text(doc_id)
                if ft:
                    compressed = compress_whitespace(ft)
                    evidence_text += f"\n=== 文档 {doc_id} (全文) ===\n{compressed}\n"
        else:
            evidence_text = self._retrieve_merged_evidence(
                q_text, options, doc_ids, card_hints, answer_format)

        # 按题型选 prompt
        if answer_format == "tf":
            prompt = TF_COT_PROMPT.format(
                evidence=evidence_text,
                question=q_text,
            )
        elif answer_format == "multi":
            prompt = MULTI_COT_PROMPT.format(
                evidence=evidence_text,
                question=q_text,
                options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
                option_a=options.get("A", ""),
                option_b=options.get("B", ""),
                option_c=options.get("C", ""),
                option_d=options.get("D", ""),
            )
        else:  # mcq
            prompt = MCQ_COT_PROMPT.format(
                evidence=evidence_text,
                question=q_text,
                options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
                option_a=options.get("A", ""),
                option_b=options.get("B", ""),
                option_c=options.get("C", ""),
                option_d=options.get("D", ""),
            )

        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "")

        try:
            result = self.qwen.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, max_tokens=4096, timeout=180,
            )
            raw_response = result["content"]
        except Exception as e:
            print(f" [ERR:{e}]")
            raw_response = ""

        answer = extract_answer_from_response(raw_response, answer_format)

        # 不做 self-critique (V14 prompt 已经强制保守)
        answer = self._post_process(answer, answer_format)
        self.memory.questions_answered += 1

        self.cot_trails.append({
            "qid": qid, "domain": domain,
            "answer": answer, "answer_format": answer_format,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": is_full_doc,
            "raw_response": raw_response,  # 不截断, 完整保留
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence_text),
            "total_doc_chars": total_doc_chars,
        }
