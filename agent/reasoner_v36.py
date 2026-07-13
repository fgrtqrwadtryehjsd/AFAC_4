"""V36: multi 自洽投票 — 采样 N 次选项级聚合, 攻击过选+早停.

## 动机 (SOP 没列, 但性价比最高的涨分动作)
multi 是 65/100 题, 完全匹配评分 (错一字母=0分). V31 60% Acc 的主要损失来自:
- 过选 (V13-fix 退分主因: multi 扩张 18→25 三字母)
- 早停 (选了 ABC 不看 D)
- A 偏置 (模糊时倾向 A)

单次采样 (temp=0.1) 无法区分"模型确信"与"模型瞎猜".
Self-Consistency (Wang 2022, GSM8K +17.9%) 证明多次采样取一致能提推理 Acc.

## 机制 (论文映射)
- Self-Consistency: 采样 N 条推理路径, 选项级聚合
- 交集投票 (防过选): 只保留 N 次都选的选项 — 过选是 V31 主损失
- 多数票兜底 (防早停): N 次答案不一致时, 取 ≥majority 次被选的选项

## 投票聚合逻辑 (零 LLM, 可离线单测)
N=3, threshold = ceil(N/2) = 2:
  - 3 次全一致 → 直接采用 (高置信)
  - 不一致 → 每选项被选次数 ≥2 才保留 (多数票, 攻击早停漏选)
  - 全都不一致且无选项≥2 → 取出现最多的那次完整答案 (退化为标准 self-consistency)

## Token 成本
multi 65 题 × N=3 × ~3K completion = ~585K 额外 token (V31 是 3.19M, +18%).
token 还有余量 (5M 预算, V31 用 3.19M). Acc 涨 2-3pp 即回本 (Acc 权重 70).

## 不自动跑全量
先零 token 单测聚合逻辑, 再小样本 (20题) LLM 验证需用户同意.
"""
import os, json
from collections import Counter
from agent.reasoner_v35 import ReasoningAgentV35, build_evidence_v35
from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_MULTI
from agent.postprocessor import extract_answer_from_response
from agent.config import RESULTS_DIR


def aggregate_multi_votes(answers: list, threshold: int = None) -> tuple:
    """multi 投票聚合 — 完整答案多数票为主 + 选项级补回漏选.

    策略 (修正: 纯选项级 threshold=2 在 N=3 全分歧时会过选, 如 ABC/ABD/ACD→ABCD):
    1. 全一致 → unanimous 直接采用
    2. 有完整答案出现 ≥ majority 次 (N=3 即 ≥2) → 采用该完整答案 (标准 self-consistency)
       这天然防过选: 完整答案 ABC 出现2次, 不会因 D 在另1次里出现就加 D
    3. 完整答案无多数 (全不同) → 选项级聚合, 但用严格 threshold=N (全一致才保留),
       防止 ABC/ABD/ACD 这种全分歧过选成 ABCD. 无选项全一致则取最常见完整答案.

    Args:
        answers: list[str], N 次采样的答案
        threshold: 未使用 (保留接口兼容), 内部按策略自适应

    Returns:
        (final_answer, confidence): unanimous / majority / fallback
    """
    valid = [a for a in answers if a and a.strip()]
    if not valid:
        return "", "fallback"

    n = len(valid)
    majority = (n + 1) // 2  # ceil(n/2): n=3→2

    # 1. 全一致
    if len(set(valid)) == 1:
        return valid[0], "unanimous"

    # 2. 完整答案多数票 (标准 self-consistency, 防过选)
    cnt = Counter(valid)
    top_answer, top_count = cnt.most_common(1)[0]
    if top_count >= majority:
        return top_answer, "majority"

    # 3. 完整答案全不同 (无多数) → 选项级严格聚合 (threshold=n, 全一致才保留)
    option_counts = Counter()
    for a in valid:
        for ch in set(a):
            option_counts[ch] += 1
    kept = sorted([ch for ch, c in option_counts.items() if c >= n])
    if kept:
        return "".join(kept), "majority"

    # 4. 全退化: 无选项全一致, 取最常见完整答案 (至少有1个)
    return top_answer, "fallback"


class ReasoningAgentV36(ReasoningAgentV35):
    """V36: V35 + multi 自洽投票 (N=3 采样, 选项级聚合).

    只改 multi 路径:
    - multi: 采样 N=3 次 (temp=0.7 多样性), 选项级聚合投票
    - tf/mcq: 完全复用 V31 (V30 精炼, 单次)
    - prompt: 完全复用 V20 (PROMPT_MULTI)
    - 后处理: 复用 V20 (但投票后已稳定, fallback A 基本不触发)
    - 证据: 完全复用 V35 (multi 上限 60K, head+锚词)
    """

    MULTI_SAMPLE_N = 3
    MULTI_SAMPLE_TEMP = 0.7  # 提高温度获得多样性 (V31 默认 0.1)

    def answer_question(self, question: dict) -> dict:
        answer_format = question.get("answer_format", "mcq")
        if answer_format == "multi":
            return self._answer_multi_v36(question)
        # tf/mcq 复用 V31 (V35 已转发到 V31)
        return ReasoningAgentV35.answer_question(self, question)

    def _answer_multi_v36(self, question: dict) -> dict:
        """V36 multi: N 次采样 + 选项级投票聚合."""
        qid = question["qid"]
        domain = question["domain"]
        q_text = question["question"]
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])

        total_doc_chars = sum(self.doc_index.doc_lengths.get(d, 0) for d in doc_ids)

        # 证据只构造一次 (复用 V35, 不重复)
        evidence = build_evidence_v35(
            self.doc_index, domain, doc_ids, self.MULTI_MAX_EVIDENCE)

        prompt = PROMPT_MULTI.format(
            evidence=evidence, question=q_text,
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
        )
        system = DOMAIN_SYSTEM.get(domain, "")

        # N 次采样
        raw_answers = []
        raw_responses = []
        for s in range(self.MULTI_SAMPLE_N):
            try:
                result = self.qwen.chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
                    temperature=self.MULTI_SAMPLE_TEMP,
                    max_tokens=4096, timeout=180,
                )
                raw = result["content"]
            except Exception as e:
                print(f" [ERR:{e}]")
                raw = ""
            raw_responses.append(raw)
            ans = extract_answer_from_response(raw, "multi")
            ans = self._post_process(ans, "multi")
            raw_answers.append(ans)

        # 选项级投票聚合
        answer, confidence = aggregate_multi_votes(raw_answers)

        # 全退化兜底 (投票都没结果, 单取第一次 post_process fallback)
        if not answer:
            answer = raw_answers[0] if raw_answers else "A"

        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": answer,
            "answer_format": "multi",
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
            "is_full_doc": False,
            "raw_answers": raw_answers,        # N 次采样结果
            "confidence": confidence,           # unanimous/majority/fallback
            "raw_response": raw_responses[0] if raw_responses else "",
            "strategy": "v36_multi_vote",
        })

        return {
            "qid": qid, "answer": answer,
            "evidence_chars": len(evidence),
            "total_doc_chars": total_doc_chars,
        }

    def save_cot_trails(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "eval_results_v36.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = []
        for t in self.cot_trails:
            t2 = dict(t)
            if "raw_response" in t2:
                t2["raw_response"] = t2["raw_response"][:1500]
            out.append(t2)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  V36 COT trails -> {path}")
