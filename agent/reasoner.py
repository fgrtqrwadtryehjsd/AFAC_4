"""推理引擎 V3

核心改进（V2→V3）：
1. 条款精准定位：题目提到"第47条"直接定位
2. 两阶段推理：先提取关键事实→再基于事实判断
3. 自校验机制：对答案做二次确认
4. 领域感知检索策略
"""
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.clause_locator import extract_clause_refs, enrich_evidence_with_clauses
from agent.postprocessor import extract_answer_from_response


# ============ 领域专用系统提示 ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": """你是一位资深保险精算师和条款解读专家。严格依据保险条款文本回答问题。

关键规则：
1. 身故保险金计算：必须严格按条款公式计算，注意"已交保费"、"现金价值"、"账户价值"、"基本保额"的区别
2. 退保金额 = 现金价值 - 退保费用（如有），注意退保费用比例随保单年度变化
3. 保险责任范围：注意"免责条款"、"等待期"、"犹豫期"等限制条件
4. 多产品比较时，必须分别计算后排序
5. 不得用常识推断，必须引用条款原文""",

    "regulatory": """你是一位金融监管合规专家。严格依据法规条文回答问题。

关键规则：
1. 法规效力层级：法律 > 行政法规 > 部门规章 > 规范性文件
2. "必须经股东大会审议"≠"须经股东大会特别决议通过"，这是不同的表决要求
3. "应当"、"必须"、"不得"=强制性；"可以"=授权性
4. 修改章程=必须经股东大会特别决议通过（2/3以上表决权）
5. 判断合规性必须逐条对照法规原文，不得用常识替代
6. 新法与旧法冲突时，适用新法；特别规定与一般规定冲突时，适用特别规定""",

    "financial_contracts": """你是一位金融合同分析专家。严格依据合同条款文本回答问题。

关键规则：
1. 债券条款：注意发行规模、利率、期限、担保方式、偿付顺序
2. 权利义务关系：区分发行人、受托管理人、持有人的权责
3. 评级信息：关注主体评级和债项评级的区别
4. 触发事件：注意违约定义、交叉违约条款、加速到期条款
5. 不得推断合同未明确约定的内容""",

    "financial_reports": """你是一位财务报表分析专家。严格依据年报数据回答问题。

关键规则：
1. 数值比较：必须精确到年报披露的数值，不得四舍五入或估算
2. 同比=和去年同期比；环比=和上期比
3. 分红政策："拟派发"（预案）≠"已派发"（实际）
4. 现金流：区分"经营活动"、"投资活动"、"筹资活动"
5. 占比计算：注意分子分母的对应关系
6. 跨年对比确保同口径""",

    "research": """你是一位行业研报分析专家。严格依据研报内容回答问题。

关键规则：
1. 行业趋势判断：必须基于研报数据，不得用外部知识
2. 公司比较：必须按研报提供的指标和口径
3. 研究结论核验：注意结论的数据支撑是否充分
4. 预测性陈述：区分"预计/预期"和"实际"数据
5. 图表数据优先于文字描述""",
}

# ============ 选项逐项验证 Prompt ============

OPTION_VERIFY_PROMPT = """## 任务
基于文档证据，逐项判断每个选项的正确性。

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 要求
对每个选项逐一分析：
- 引用文档原文中的关键语句作为依据
- 明确判断该选项"正确"或"错误"
- 多选题中，每个选项必须独立判断

严格按以下格式输出：
选项A：[正确/错误] — 依据：[引用文档证据]
选项B：[正确/错误] — 依据：[引用文档证据]
选项C：[正确/错误] — 依据：[引用文档证据]
选项D：[正确/错误] — 依据：[引用文档证据]

最终答案：{answer_hint}"""

# ============ 两阶段推理 - 事实提取 ============

FACT_EXTRACT_PROMPT = """基于以下文档证据，提取与问题直接相关的关键事实。

文档证据：
{evidence}

问题：{question}
关注的方面：{focus}

请只输出关键事实（数字、条款、条件、触发规则等），不要解释，不要推理："""

# ============ 两阶段推理 - 基于事实判断 ============

FACT_JUDGE_PROMPT = """基于提取的关键事实，判断问题各选项的正确性。

关键事实：
{facts}

问题：{question}

选项：
{options}

请逐项判断，格式如下：
选项A：[正确/错误] — 依据：[引用关键事实]
选项B：[正确/错误] — 依据：[引用关键事实]
选项C：[正确/错误] — 依据：[引用关键事实]
选项D：[正确/错误] — 依据：[引用关键事实]

最终答案：{answer_hint}"""

# ============ 自校验 Prompt ============

SELF_CHECK_PROMPT = """请校验以下答案的正确性。你必须严格基于文档证据，不能添加没有证据支持的选项。

问题：{question}
选项：
{options}

已给出的答案：{answer}

文档证据：
{evidence}

校验规则：
1. 只有文档证据明确支持的选项才能选
2. 如果某个选项在证据中找不到充分支持，应当排除
3. 宁可少选也不要多选（多选错选均计为错误）

校验结果：[确认正确/应纠正为X]
如需纠正，给出正确答案："""


# ============ 辅助函数 ============

def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options.keys()))

def format_evidence(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        src = chunk.get("doc_id", "?")
        typ = chunk.get("type", "")
        tag = f"来源:{src}" + (f", {typ}" if typ else "")
        parts.append(f"### 证据 {i}（{tag}）\n{chunk['text']}")
    return "\n\n".join(parts)

def get_answer_hint(answer_format: str) -> str:
    return {"mcq": "一个大写字母(A/B/C/D)", "multi": "多个大写字母按字母序(如ABC)", "tf": "A或B"}.get(answer_format, "一个大写字母")

def get_domain_focus(domain: str) -> str:
    return {
        "insurance": "保险责任、身故保险金、退保金额、领取规则、计算公式",
        "regulatory": "法规条款、适用范围、合规义务、时限、处罚规定",
        "financial_contracts": "债券条款、发行信息、评级、权利义务",
        "financial_reports": "营业收入、净利润、现金流、分红、研发投入",
        "research": "行业趋势、公司指标、研究结论、预测数据",
    }.get(domain, "关键数字和条款")


# ============ 推理 Agent V3 ============

class ReasoningAgent:
    """推理 Agent V3 — 条款精准定位 + 两阶段推理 + 自校验"""

    def __init__(self, qwen_client: QwenClient, doc_index: DocumentIndex, token_budget: int = 4_000_000):
        self.qwen = qwen_client
        self.index = doc_index
        self.token_budget = token_budget

    def tokens_remaining(self) -> int:
        return max(0, self.token_budget - self.qwen.get_token_stats()["total_tokens"])

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        question_text = question["question"]
        options = question["options"]
        answer_format = question["answer_format"]
        doc_ids = question.get("doc_ids", [])

        print(f"  {qid} ({domain}/{answer_format})", end=" ")

        # Step 1: 检索证据（含条款精准定位）
        chunks = self._smart_retrieve(question_text, options, doc_ids, domain)

        # Step 2: 压缩证据
        evidence = self._compress_evidence(chunks, max_chars=12000)

        # Step 3: 推理（两阶段 or 直接）
        if answer_format == "multi" and self.tokens_remaining() > 30000:
            answer, raw = self._two_stage_reason(domain, question_text, options, evidence, answer_format)
        else:
            answer, raw = self._direct_reason(domain, question_text, options, evidence, answer_format)

        # Step 4: 答案提取
        if not answer:
            answer = extract_answer_from_response(raw, answer_format)

        # Step 5: 自校验（仅对高风险题型，token充裕时）
        if answer and answer_format == "multi" and self.tokens_remaining() > 15000:
            answer, raw = self._self_check(question_text, options, answer, evidence, answer_format, raw)

        print(f"→ {answer}")
        return {
            "qid": qid, "answer": answer, "raw_response": raw,
            "evidence_chunks": chunks,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _smart_retrieve(self, question: str, options: dict, doc_ids: list, domain: str) -> list:
        """智能检索：BM25 + 条款精准定位"""
        # BM25 检索
        if doc_ids:
            chunks = self.index.search_with_rerank(question, doc_ids=doc_ids, top_k=5)
            for opt_key, opt_text in options.items():
                for c in self.index.search_with_rerank(f"{question} {opt_text}", doc_ids=doc_ids, top_k=2):
                    if c not in chunks:
                        chunks.append(c)
            # 如果检索不足，直接取文档条款块
            if len(chunks) < 3:
                for doc_id in doc_ids:
                    for c in self.index.get_doc_chunks(doc_id)[:3]:
                        if c not in chunks:
                            chunks.append(c)
        else:
            chunks = self.index.search_with_rerank(question, top_k=5)
            for opt_key, opt_text in options.items():
                for c in self.index.search_with_rerank(f"{question} {opt_text}", top_k=2):
                    if c not in chunks:
                        chunks.append(c)

        # 条款精准定位（监管/合同领域关键优化）
        if domain in ("regulatory", "financial_contracts"):
            clause_refs = extract_clause_refs(question, options)
            if clause_refs and doc_ids:
                extra = enrich_evidence_with_clauses(self.index, doc_ids, clause_refs)
                # 去重并优先放入
                for ec in extra:
                    if ec not in chunks:
                        chunks.insert(0, ec)  # 精准条款放最前面

        return chunks

    def _compress_evidence(self, chunks: list, max_chars: int = 12000) -> str:
        total = sum(len(c["text"]) for c in chunks)
        if total <= max_chars:
            return format_evidence(chunks)
        per_chunk = max(500, max_chars // max(len(chunks), 1))
        compressed = [{**c, "text": c["text"][:per_chunk] + ("..." if len(c["text"]) > per_chunk else "")} for c in chunks]
        return format_evidence(compressed)

    def _direct_reason(self, domain: str, question: str, options: dict, evidence: str, answer_format: str) -> tuple:
        """直接推理（单选/判断题）"""
        sys_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "你是一个专业的金融文档分析助手。严格依据证据回答。")
        user_prompt = OPTION_VERIFY_PROMPT.format(
            evidence=evidence, question=question,
            options=format_options(options), answer_hint=get_answer_hint(answer_format),
        )
        result = self.qwen.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1, max_tokens=2048,
        )
        answer = extract_answer_from_response(result["content"], answer_format)
        return answer, result["content"]

    def _two_stage_reason(self, domain: str, question: str, options: dict, evidence: str, answer_format: str) -> tuple:
        """两阶段推理（多选题关键策略）：先提取事实→再判断选项"""
        sys_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "你是一个专业的金融文档分析助手。严格依据证据回答。")

        # Stage 1: 提取关键事实
        fact_prompt = FACT_EXTRACT_PROMPT.format(
            evidence=evidence, question=question, focus=get_domain_focus(domain),
        )
        fact_result = self.qwen.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": fact_prompt}],
            temperature=0.0, max_tokens=1024,
        )
        facts = fact_result["content"]

        # Stage 2: 基于事实判断
        judge_prompt = FACT_JUDGE_PROMPT.format(
            facts=facts, question=question,
            options=format_options(options), answer_hint=get_answer_hint(answer_format),
        )
        judge_result = self.qwen.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": judge_prompt}],
            temperature=0.1, max_tokens=2048,
        )
        answer = extract_answer_from_response(judge_result["content"], answer_format)
        return answer, f"[事实提取]\n{facts}\n\n[选项判断]\n{judge_result['content']}"

    def _self_check(self, question: str, options: dict, answer: str, evidence: str, answer_format: str, prev_raw: str) -> tuple:
        """自校验：对答案做二次确认"""
        check_prompt = SELF_CHECK_PROMPT.format(
            question=question, options=format_options(options),
            answer=answer, evidence=evidence[:4000],
        )
        check_result = self.qwen.chat(
            [{"role": "user", "content": check_prompt}],
            temperature=0.0, max_tokens=256,
        )
        content = check_result["content"]

        # 如果校验建议纠正
        if "应纠正" in content:
            corrected = extract_answer_from_response(content, answer_format)
            if corrected and corrected != answer:
                print(f"[校验纠正: {answer}→{corrected}]", end=" ")
                return corrected, prev_raw + f"\n[自校验纠正: {answer}→{corrected}]"

        return answer, prev_raw
