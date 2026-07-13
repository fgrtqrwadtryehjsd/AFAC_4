"""V41 创新点②: 上下文手术刀 — 倒U型重排 (LongContextReorder)

## 用户创新点②原文
不把召回文档分块直接丢给大模型. 检索后插入抽取层, 只输出"数值/实体/财报项",
使用倒U型重排: 最重要金融数据片段放 Context 最前和最后, 中间放次要信息,
对抗 LLM "中间迷失"效应.

## V41 实现
- 抽取层 = 创新点③的三档分层 (保留原貌段落, 非碎片)
- 倒U重排: 作用于完整段落 (TIER_KEEP), 重要放首尾, 次要摘要放中间
- 关键修正: V40 的碎片化是失败根因, V41 重排对象是完整段落

## 零 token
纯列表重排.
"""


def u_shaped_reorder_segments(segments):
    """倒U型重排: 高分段落放首尾, 低分放中间.

    Args:
        segments: list of {tier, doc_id, score, text, tag}, 已按 score 降序

    Returns:
        重排后的 segments 列表: [最高, 第3高, ..., 最低, ..., 第4高, 第2高]
        首尾都是高分段落, 对抗中间迷失.
    """
    if len(segments) <= 2:
        return list(segments)

    # 按 score 降序
    sorted_segs = sorted(segments, key=lambda x: -x.get("score", 0))

    front = []  # 首部: 奇数位 (0,2,4,...) 高分
    back = []   # 尾部: 偶数位 (1,3,5,...) 次高分, 反转后尾部是高分
    for i, s in enumerate(sorted_segs):
        if i % 2 == 0:
            front.append(s)
        else:
            back.append(s)
    back.reverse()
    return front + back


def render_evidence(segments):
    """把重排后的段落渲染成证据文本.

    每段标注 doc_id + 命中标签, 帮助模型定位 (创新点①状态锚定配合).
    """
    parts = []
    for s in segments:
        did = s["doc_id"]
        tag = s.get("tag", "")
        text = s["text"]
        header = f"=== {did}"
        if tag and tag != "summary":
            header += f" [命中:{tag}]"
        header += " ==="
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)
