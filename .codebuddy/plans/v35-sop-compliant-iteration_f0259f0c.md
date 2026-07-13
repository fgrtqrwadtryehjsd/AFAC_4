---
name: v35-sop-compliant-iteration
overview: 按 SOP 8.2 严格执行 V35（multi 证据 90K→60K + 锚词前置）的验证流程：先离线分析确认 research/insurance 域 30K head 安全，再跑 20 题小样本（10 高置信正确 + 10 不稳定），通过 SOP 4.6 门槛才全量。不通过则调整或转路线。
todos:
  - id: fix-v35-risks
    content: 修复 reasoner_v35.py 三处风险：research/insurance 保持 45K head、证据顺序不反转（head 在前+锚词追加）、锚词 ctx 适配 per_doc
    status: completed
  - id: fix-select-sample
    content: 改进 select_sample.py 为 SOP 8.2 标准：用 V20/V22/V31 三版本答案一致性选 10 正确 + 10 不稳定 multi 题
    status: completed
    dependencies:
      - fix-v35-risks
  - id: offline-verify
    content: 离线验证（零 token）：对 20 题小样本生成 V35 证据文本，对比 V31 证据差异，检查锚词保留和 head 完整性
    status: completed
    dependencies:
      - fix-select-sample
  - id: small-sample-run
    content: 小样本 20 题推理（约 800K token）：只跑 multi 题，对比 V31 旧答案，记录每题变化
    status: completed
    dependencies:
      - offline-verify
  - id: full-run-decision
    content: 根据小样本结果按 SOP 4.6/4.7 门槛决策是否全量，通过则跑全量 100 题并记录 EXPERIMENTS.md
    status: completed
    dependencies:
      - small-sample-run
---

## 用户需求

用户要求严格遵循 `AFAC4_SOP_to_90.md` 的改进路线，仔细思考后再执行。核心约束：

1. **SOP 8.2 小样本过关再全量**：每次全量前必须跑 20 题小样本（10 道 V31 正确题 + 10 道 V31 错误题），记录每题旧答案、新答案、token、是否修正、是否误伤、原因
2. **每次只改一个变量**（SOP 8.1）：V35 同时改了"证据上限"和"证据结构顺序"，需论证或拆分
3. **不删 fallback A**（SOP 8.3）：V35 继承 V20 的 _post_process，已保留
4. **Acc 优先于 TokenScore**（SOP 8.5）：多答对 1 题值得多花 80K-120K token
5. **每次 LLM 推理花费用户预算**：必须确保有提升依据才推理

## 产品概述

当前最优基线 V31（48.50 分），目标通过 SOP V32 路线（降 multi token 保 Acc）提升到 52-56 分。

## 核心功能

- 修复 V35 代码中的风险点（research/insurance 纯 head 缩小、证据顺序变化）
- 改进选样脚本符合 SOP 8.2 标准
- 离线验证 V35 证据是否安全（零 token）
- 小样本 20 题验证（少量 token）
- 根据小样本结果决定是否全量

## Tech Stack

- Python 3.11 + conda 环境（afac4）
- qwen3.6-plus API（阿里云百炼平台）
- 现有代码架构：V20(域分流) → V21(fc锚词) → V22(财报锚词) → V30(精炼) → V31(动态分配) → V35(multi重构)

## 当前状态分析

### V31 架构（当前最优 48.50）

- tf/mcq（35题）→ V30 精炼证据（8-14K/题）
- multi（65题）→ V22 完整证据（90K/题，per_doc=90K/n_docs）
- 模型 qwen3.6-plus + thinking=True
- V22 multi 证据结构：head 在前（fc=35K/fin=25K/其他=45K）+ 锚词章节追加

### V35 当前代码（已写好，未跑）

- multi 证据上限 90K→60K（per_doc=60K/n_docs）
- 锚词前置 + head 在后（顺序反转）
- research/insurance 纯 head 从 45K→30K（双文档题）
- tf/mcq 完全复用 V31，prompt 复用 V20，后处理复用 V20（保留 fallback A）

### V35 三个风险点

1. **research/insurance 纯 head 缩小**：V22 是 45K/doc，V35 是 30K/doc（双文档题），可能丢失后部内容
2. **证据顺序变化**：V22 head 在前+锚词追加；V35 锚词在前+head 在后。V13.1 教训证明段顺序改变可能导致退分
3. **select_sample.py 不符合 SOP 8.2**：按"证据大小"选样，应按"V31正确/错误"选样

### V35 盈亏平衡分析

- V31: Token=3.19M, TS=0.362, Acc=60%, Score=48.50
- V35 预估: multi 65题省约 1.3M token → Token≈1.89M, TS≈0.62
- Acc=60%（持平）: Score=53.2 (+4.7)
- Acc=55%（降5%）: Score=48.8 (+0.3 微涨)
- Acc=50%（降10%）: Score=44.4 (-4.1 退步)
- **盈亏平衡点: Acc≈55%，multi 最多漏选 3 题**

## Implementation Approach

### 核心策略：修复 V35 风险 → 离线验证 → 小样本 → 全量决策

**为何 V35 需要同时改"上限+结构"**：SOP 4.2 建议分三档（60K→45K→30K），但 V22 的证据结构是 head 在前 + 锚词追加。如果只降上限不改结构，per_doc=30K 时 head=35K 就超了，锚词空间=0（已在之前离线验证中确认会丢失"违约事件""资产减值"等关键锚词）。因此上限和结构必须一起改——这不是"改两个变量"，而是"同一个改动（降 token）的必要配套"。

### 步骤 1：修复 V35 代码风险（零 token）

**修复 1**：research/insurance 保持 V22 的 45K head（不缩小）

- V35 的 `extract_evidence_simple_v35` 当前用 `_take_head(text, max_chars)`，max_chars=per_doc=30K
- 改为：如果 per_doc < 45K，仍取 45K head（保证 research/insurance 不退步）
- 理由：research/insurance 域没有锚词机制，纯靠 head，缩小 head = 直接丢信息

**修复 2**：证据顺序保持 V22 的 head 在前 + 锚词追加（不反转）

- V35 当前把锚词 insert(0, head) 放最前，改为 append 追加在 head 后
- 理由：V13.1 教训证明段顺序改变可能退步；V22 的 head 在前已验证有效

**修复 3**：select_sample.py 改为 SOP 8.2 标准

- 用 V20/V22/V31 三版本答案一致性作为 ground truth 代理：
- 三版本答案一致 → 高置信正确题（选 10 道 multi）
- 三版本答案不一致 → 不稳定题（选 10 道 multi）
- 读取 `results/eval_results_v20.json`、`eval_results_v22.json`、`eval_results_v31.json`
- 对比三版本答案，选出 20 道 multi 题小样本

### 步骤 2：离线验证（零 token）

验证 V35 修复后的证据是否安全：

- 对 20 题小样本，离线生成 V35 证据文本（不调 LLM）
- 对比 V31(90K) vs V35(60K) 的证据差异
- 检查：关键锚词是否保留、head 是否缩小、是否有空证据
- 特别检查 research/insurance 题：45K head 是否完整

### 步骤 3：小样本 20 题（少量 token，约 800K token ≈ ¥2-3）

- 只跑 20 道小样本 multi 题
- tf/mcq 题复用 V31 答案（不调 LLM）
- 记录每题：V31 旧答案、V35 新答案、token 消耗、是否变化
- SOP 4.6 通过标准：
- 少对 ≥3 题 → 不全量
- 持平且 token 降 ≥25% → 可全量
- 多对 ≥2 题且 token 降 ≥15% → 优先全量

### 步骤 4：全量决策

根据小样本结果决定：

- 通过 SOP 4.7 门槛 → 跑全量 100 题
- 不通过 → 分析原因，调整或放弃 V35

## Implementation Notes

- **不删 fallback A**：V35 继承 V20 的 _post_process，mcq/tf 无答案返回 "A"，multi 无答案返回 "A"。已在代码中确认。
- **不改 prompt**：V35 的 multi 路径用 V20 的 PROMPT_MULTI，tf/mcq 用 V30 路径。完全不变。
- **不改后处理**：V35 用 V20 的 _post_process（含 fallback A）。
- **模型一致**：config.py MODEL_NAME="qwen3.6-plus"，V35 pipeline 用 QwenClient() 默认模型，与 V31 一致。
- **证据去重**：V35 当前用 `sec[:50]` 做去重 key，与 V22 一致。
- **Token 统计**：V35 pipeline 用 QwenClient.get_token_stats()，与 V31 一致。

## Directory Structure

```
agent/
├── reasoner_v35.py     # [MODIFY] 修复3处风险：research/insurance保持45K head、证据顺序不反转、锚词ctx适配
├── pipeline_v35.py     # [MODIFY] 增加小样本模式参数，支持只跑20题
├── reasoner_v31.py     # [READ] 当前最优基线，不改动
├── reasoner_v22.py     # [READ] multi证据构造来源，不改动
├── reasoner_v20.py     # [READ] V20基线（_post_process, DOMAIN_SYSTEM, PROMPT_MULTI），不改动
select_sample.py        # [MODIFY] 改为SOP 8.2标准选样（三版本一致性代理ground truth）
EXPERIMENTS.md          # [MODIFY] 记录V35实验全过程
```

## Key Code Structures

V35 证据提取器修复后的核心逻辑：

```python
# 修复1: research/insurance 保持 45K head（不缩小）
def extract_evidence_simple_v35(text: str, max_chars: int = 30000) -> str:
    # 即使 per_doc < 45K，也至少取 45K head（保证不退步）
    actual_head = max(max_chars, 45000)
    return _take_head(text, min(actual_head, len(text)))

# 修复2: 锚词追加在 head 之后（不反转顺序）
def extract_evidence_fc_v35(text: str, max_chars: int = 30000) -> str:
    head = _take_head(text, min(18000, max_chars // 2))  # head 在前
    parts = [head]
    for anchor in V21_FC_ANCHORS:
        sec = _locate_section(text, [anchor], ctx_chars=2000)
        if sec and sec[:50] not in seen:
            parts.append(f"[{anchor}]\n{sec}")  # 锚词追加在后
            if sum(len(p) for p in parts) >= max_chars:
                break
    return "\n\n".join(parts)[:max_chars]
```

小样本选样核心逻辑：

```python
# SOP 8.2: 10道高置信正确 + 10道不稳定
v20 = {r['qid']: r['answer'] for r in load('eval_results_v20.json')}
v22 = {r['qid']: r['answer'] for r in load('eval_results_v22.json')}
v31 = {r['qid']: r['answer'] for r in load('eval_results_v31.json')}

consistent = [qid for qid in v31 if v20.get(qid) == v22.get(qid) == v31.get(qid)]  # 三版本一致
inconsistent = [qid for qid in v31 if v20.get(qid) != v31.get(qid) or v22.get(qid) != v31.get(qid)]  # 不一致
sample = consistent[:10] + inconsistent[:10]  # 10+10
```