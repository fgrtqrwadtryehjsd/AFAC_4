"""V41 创新点①: 锚定增量状态机 (Anchored Incremental State)

## 用户创新点①原文
递进式提问场景, 每次交互后用小模型维护结构化全局状态字典:
{
  "当前分析标的": "某上市公司",
  "已提取核心财务数据": {"2023营收": "X亿", ...},
  "用户倾向": "横向竞品对比"
}
只带精简 JSON 字典 + 当前问题, 摒弃冗长历史对话树.

## 赛题适配 (单题独立, 无多轮递进)
"增量/跨轮"在赛题不存在, 但"结构化状态代替长文本"的核心思想落地为:
每题构建锚定状态字典, 作为元信息注入 prompt:
  - 分析标的: 从 doc_ids 推断 (公司/产品/法规名)
  - 题型意图: multi/mcq/tf + 比较型/计算型/核验型
  - 选项 claim 分解: 每选项的关键词+数字 (来自创新点③的 claim 分解)
  - 比较维度: 跨文档对比题标注 "需对比 doc1 vs doc2"

状态字典让模型先"锚定"要查什么, 再看证据, 而非在长证据里迷失.

## 零 token
纯规则构建状态字典.
"""
import re
from agent.memory_compressor_v41 import ImportanceScorer


class AnchorState:
    """锚定状态机: 构建单题结构化状态字典."""

    def __init__(self):
        self.scorer = ImportanceScorer()

    def build(self, question, doc_index):
        """构建锚定状态字典.

        Returns:
            state: dict, 结构化状态
            state_text: str, 渲染成 prompt 可读文本
        """
        qid = question.get("qid", "")
        domain = question.get("domain", "")
        q_text = question.get("question", "")
        answer_format = question.get("answer_format", "mcq")
        doc_ids = question.get("doc_ids", [])

        # 1. 分析标的: 从 doc_ids 推断标的类型
        targets = self._infer_targets(doc_ids, domain, doc_index)

        # 2. 题型意图
        intent = self._infer_intent(q_text, answer_format, doc_ids)

        # 3. 选项 claim 分解 (来自创新点③)
        claims = self.scorer.decompose_option_claims(question)

        # 4. 比较维度
        compare_dim = self._infer_compare_dimension(q_text, doc_ids)

        state = {
            "分析标的": targets,
            "题型意图": intent,
            "选项claim": {k: {"关键词": v["keywords"], "数字": v["numbers"]}
                          for k, v in claims.items()},
            "比较维度": compare_dim,
        }

        state_text = self._render(state)
        return state, state_text

    def _infer_targets(self, doc_ids, domain, doc_index):
        """从 doc_ids 推断分析标的."""
        targets = []
        for did in doc_ids:
            # doc_id 本身常含公司/产品名 (annual_byd_2024_report)
            name = did
            if did.startswith("annual_"):
                # annual_byd_2024_report → 比亚迪 2024年报
                parts = did.replace("annual_", "").replace("_report", "").split("_")
                if parts:
                    name = f"{parts[0]} {parts[1] if len(parts)>1 else ''}年报".strip()
            elif did.startswith("pack2_text"):
                name = f"研报{did}"
            targets.append(name)
        return targets

    def _infer_intent(self, q_text, answer_format, doc_ids):
        """推断题型意图."""
        intent = answer_format
        if any(w in q_text for w in ["对比", "比较", "分别", "两份", "两年"]):
            intent += "-跨文档对比"
        if any(w in q_text for w in ["计算", "多少", "占比", "比例", "增长率"]):
            intent += "-数值计算"
        if any(w in q_text for w in ["是否", "正确", "准确", "符合"]):
            intent += "-事实核验"
        if len(doc_ids) >= 2:
            intent += f"({len(doc_ids)}文档)"
        return intent

    def _infer_compare_dimension(self, q_text, doc_ids):
        """推断比较维度 (跨文档题标注)."""
        dims = []
        if any(w in q_text for w in ["连续两年", "2024", "2025", "同比"]):
            dims.append("年度对比")
        if any(w in q_text for w in ["两份", "对比", "分别"]):
            dims.append("文档间对比")
        return dims

    def _render(self, state):
        """渲染状态字典为 prompt 可读文本."""
        lines = ["## 锚定状态 (Anchored State)"]
        lines.append(f"- 分析标的: {', '.join(state['分析标的']) if state['分析标的'] else '未知'}")
        lines.append(f"- 题型意图: {state['题型意图']}")
        if state["比较维度"]:
            lines.append(f"- 比较维度: {', '.join(state['比较维度'])}")
        lines.append("- 选项核验要点:")
        for ok, c in state["选项claim"].items():
            kws = ",".join(c["关键词"]) if c["关键词"] else "无明确关键词"
            nums = ",".join(c["数字"][:3]) if c["数字"] else ""
            line = f"  选项{ok}: 关键词[{kws}]"
            if nums:
                line += f" 数字[{nums}]"
            lines.append(line)
        lines.append("")
        return "\n".join(lines)
