"""推理引擎 V2

核心改进：
1. 5领域专用推理Prompt
2. 选项逐项验证（多选题关键）
3. 两阶段推理（提取事实→判断选项）
4. Token预算管控
"""
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.postprocessor import validate_answer, extract_answer_from_response


# ============ 领域专用系统提示 ============

DOMAIN_SYSTEM_PROMPTS = {
    "insurance": """你是一位资深保险精算师和条款解读专家。你需要严格依据保险条款文本回答问题。

关键规则：
1. 身故保险金计算：必须严格按条款公式计算，注意"已交保费"、"现金价值"、"账户价值"、"基本保额"的区别
2. 退保金额 = 现金价值 - 退保费用（如有），注意退保费用比例随保单年度变化
3. 保险责任范围：注意"免责条款"、"等待期"、"犹豫期"等限制条件
4. 多产品比较时，必须分别计算后排序
5. 不得用常识推断，必须引用条款原文""",

    "regulatory": """你是一位金融监管合规专家。你需要严格依据法规条文回答问题。

关键规则：
1. 法规效力层级：法律 > 行政法规 > 部门规章 > 规范性文件
2. 必须经"股东大会审议"和"须经股东大会特别决议通过"是不同要求
3. "应当"、"必须"、"不得"表示强制性要求；"可以"表示授权性规定
4. 时限要求：注意"之日起X日内"、"届满前"等表述
5. 判断合规性时，必须逐条对照法规原文，不得用常识替代
6. 修改章程必须经股东大会特别决议通过（2/3以上表决权）""",

    "financial_contracts": """你是一位金融合同分析专家。你需要严格依据合同条款文本回答问题。

关键规则：
1. 债券条款：注意发行规模、利率、期限、担保方式、偿付顺序
2. 权利义务关系：区分发行人、受托管理人、持有人的权责
3. 评级信息：关注主体评级和债项评级的区别
4. 触发事件：注意违约定义、交叉违约条款、加速到期条款
5. 不得推断合同未明确约定的内容""",

    "financial_reports": """你是一位财务报表分析专家。你需要严格依据年报数据回答问题。

关键规则：
1. 数值比较：必须精确到年报披露的数值，不得四舍五入或估算
2. 同比/环比：注意"同比增长"是和去年同期比，"环比"是和上期比
3. 分红政策：注意"拟派发"（预案）vs"已派发"（实际）的区别
4. 现金流：区分"经营活动"、"投资活动"、"筹资活动"现金流
5. 占比计算：注意分子分母的对应关系
6. 跨年对比时确保同口径比较""",

    "research": """你是一位行业研报分析专家。你需要严格依据研报内容回答问题。

关键规则：
1. 行业趋势判断：必须基于研报数据，不得用外部知识
2. 公司比较：必须按研报提供的指标和口径比较
3. 研究结论核验：注意研报结论的数据支撑是否充分
4. 预测性陈述：区分"预计/预期"和"实际"数据
5. 研报中的图表数据优先于文字描述""",
}

# ============ 选项逐项验证 Prompt ============

OPTION_VERIFICATION_PROMPT = """## 任务
请基于以下文档证据，逐项判断每个选项的正确性。

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 要求
请对每个选项逐一分析：
- 引用文档原文中的关键语句作为依据
- 明确判断该选项"正确"或"错误"
- 注意：判断题/多选题中，每个选项必须独立判断

请严格按以下格式输出：
选项A：[正确/错误] — 依据：[引用文档证据]
选项B：[正确/错误] — 依据：[引用文档证据]
选项C：[正确/错误] — 依据：[引用文档证据]
选项D：[正确/错误] — 依据：[引用文档证据]

最终答案：{answer_hint}"""

# ============ 单选题/判断题专用 Prompt ============

MCQ_JUDGE_PROMPT = """## 任务
基于文档证据，判断哪个选项是正确答案。

## 文档证据
{evidence}

## 问题
{question}

## 选项
{options}

## 要求
1. 严格依据文档证据判断，不用常识
2. 逐项分析每个选项，说明其正确或错误的原因
3. 只有一个正确选项

请严格按以下格式输出：
选项A：[正确/错误] — 依据：[简述]
选项B：[正确/错误] — 依据：[简述]
选项C：[正确/错误] — 依据：[简述]
选项D：[正确/错误] — 依据：[简述]

最终答案：X（单个大写字母）"""

# ============ 事实提取 Prompt（两阶段推理）============

FACT_EXTRACTION_PROMPT = """从以下文档中提取与问题相关的关键事实。

文档内容：
{doc_text}

问题：{question}
关注方面：{focus_areas}

请提取关键事实（数字、条款、条件、触发规则等），只输出与问题直接相关的事实，不要解释："""

# ============ 辅助函数 ============

def format_options(options: dict) -> str:
    lines = []
    for key in sorted(options.keys()):
        lines.append(f"{key}. {options[key]}")
    return "\n".join(lines)

def format_evidence(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("doc_id", "未知文档")
        chunk_type = chunk.get("type", "")
        parts.append(f"### 证据 {i}（来源：{source}，类型：{chunk_type}）\n{chunk['text']}")
    return "\n\n".join(parts)

def get_answer_hint(answer_format: str) -> str:
    hints = {
        "mcq": "一个大写字母（A/B/C/D）",
        "multi": "多个大写字母按字母序排列（如ABC），无分隔符",
        "tf": "一个大写字母（A或B）",
    }
    return hints.get(answer_format, "一个大写字母")


# ============ 推理 Agent V2 ============

class ReasoningAgent:
    """推理 Agent V2 — 领域感知 + 选项逐项验证"""

    def __init__(self, qwen_client: QwenClient, doc_index: DocumentIndex, token_budget: int = 5_000_000):
        self.qwen = qwen_client
        self.index = doc_index
        self.token_budget = token_budget
        self.memory = {}  # doc_id -> compressed_summary
        self.doc_facts = {}  # doc_id -> extracted_facts

    def tokens_remaining(self) -> int:
        stats = self.qwen.get_token_stats()
        return max(0, self.token_budget - stats["total_tokens"])

    def answer_question(self, question: dict) -> dict:
        qid = question["qid"]
        domain = question["domain"]
        question_text = question["question"]
        options = question["options"]
        answer_format = question["answer_format"]
        doc_ids = question.get("doc_ids", [])

        print(f"  回答 {qid} ({domain}/{answer_format})...", end=" ")

        # Step 1: 检索证据
        if doc_ids:
            chunks = self._retrieve_with_doc_ids(question_text, options, doc_ids)
        else:
            chunks = self.index.search_with_rerank(question_text, top_k=5)
            # B榜补充：按选项检索
            for opt_key, opt_text in options.items():
                extra = self.index.search_with_rerank(
                    f"{question_text} {opt_text}", top_k=2
                )
                for c in extra:
                    if c not in chunks:
                        chunks.append(c)

        # Step 2: 控制证据总量（节省 token）
        evidence = self._compress_evidence(chunks, max_chars=8000)

        # Step 3: 领域专用推理
        if answer_format == "multi":
            answer, raw = self._verify_each_option(domain, question_text, options, evidence, answer_format)
        elif answer_format in ("mcq", "tf"):
            answer, raw = self._judge_single(domain, question_text, options, evidence, answer_format)
        else:
            answer, raw = self._general_reasoning(domain, question_text, options, evidence, answer_format)

        # Step 4: 答案校验
        if not answer:
            answer = extract_answer_from_response(raw, answer_format)

        print(f"答案={answer}")
        return {
            "qid": qid,
            "answer": answer,
            "raw_response": raw,
            "evidence_chunks": chunks,
            "tokens": {
                "prompt_tokens": 0,  # 由 qwen_client 累计
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def _retrieve_with_doc_ids(self, question: str, options: dict, doc_ids: list) -> list:
        """A榜精准检索：已知文档 + BM25 + Rerank"""
        chunks = self.index.search_with_rerank(question, doc_ids=doc_ids, top_k=5)
        
        # 按选项补充检索
        for opt_key, opt_text in options.items():
            query = f"{question} {opt_text}"
            extra = self.index.search_with_rerank(query, doc_ids=doc_ids, top_k=2)
            for c in extra:
                if c not in chunks:
                    chunks.append(c)

        # 如果 BM25 检索结果太少，直接取相关文档的条款块
        if len(chunks) < 3:
            for doc_id in doc_ids:
                doc_chunks = self.index.get_doc_chunks(doc_id)
                if doc_chunks:
                    # 取前几个块补充
                    for c in doc_chunks[:3]:
                        if c not in chunks:
                            chunks.append(c)

        return chunks

    def _compress_evidence(self, chunks: list, max_chars: int = 12000) -> str:
        """压缩证据：如果总量太大，截断每个块"""
        total = sum(len(c["text"]) for c in chunks)
        if total <= max_chars:
            return format_evidence(chunks)
        
        # 按块均分额度，但每个块至少保留 500 字
        per_chunk = max(500, max_chars // max(len(chunks), 1))
        compressed_chunks = []
        for c in chunks:
            compressed_chunks.append({
                **c,
                "text": c["text"][:per_chunk] + ("..." if len(c["text"]) > per_chunk else ""),
            })
        return format_evidence(compressed_chunks)

    def _verify_each_option(self, domain: str, question: str, options: dict, evidence: str, answer_format: str) -> tuple:
        """选项逐项验证（多选题关键策略）"""
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "你是一个专业的金融文档分析助手。严格依据证据回答。")
        user_prompt = OPTION_VERIFICATION_PROMPT.format(
            evidence=evidence,
            question=question,
            options=format_options(options),
            answer_hint=get_answer_hint(answer_format),
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        result = self.qwen.chat(messages, temperature=0.1, max_tokens=2048)
        content = result["content"]
        answer = extract_answer_from_response(content, answer_format)
        return answer, content

    def _judge_single(self, domain: str, question: str, options: dict, evidence: str, answer_format: str) -> tuple:
        """单选题/判断题专用推理"""
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "你是一个专业的金融文档分析助手。严格依据证据回答。")
        user_prompt = MCQ_JUDGE_PROMPT.format(
            evidence=evidence,
            question=question,
            options=format_options(options),
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        result = self.qwen.chat(messages, temperature=0.1, max_tokens=1024)
        content = result["content"]
        answer = extract_answer_from_response(content, answer_format)
        return answer, content

    def _general_reasoning(self, domain: str, question: str, options: dict, evidence: str, answer_format: str) -> tuple:
        """通用推理"""
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "你是一个专业的金融文档分析助手。严格依据证据回答。")
        user_prompt = OPTION_VERIFICATION_PROMPT.format(
            evidence=evidence,
            question=question,
            options=format_options(options),
            answer_hint=get_answer_hint(answer_format),
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        result = self.qwen.chat(messages, temperature=0.1, max_tokens=2048)
        content = result["content"]
        answer = extract_answer_from_response(content, answer_format)
        return answer, content
