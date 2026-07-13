# AFAC2026 赛题四：金融长文本Agent的动态记忆压缩与高效问答

## 1. 赛题与评分

- **任务**：给定金融长文档（保险条款/监管法规/金融合同/财报/研报）与题目，Agent 需动态压缩记忆并高效问答。题型：tf（判断）、mcq（单选）、multi（多选）。
- **评分**：`FinalScore = 100 × Accuracy × (0.7 + 0.3 × TokenScore)`，`TokenScore = max(0, min(1, (5,000,000 − TotalTokens)/5,000,000))`。Acc 权重 70 ≫ Token 权重 30。
- **约束**：推理阶段仅 Qwen 系列 API（`qwen3.6-plus`），不微调；禁止非 Qwen 模型参与 rerank/投票/纠错；预处理阶段可用 MinerU 等。

## 2. 最终方案 V43

V43 按题型分流，针对 multi 与 tf/mcq 的不同瓶颈分别优化：

### 2.1 multi（65题）：知识图谱压缩记忆 + 多轮图算法推理（V42）

**瓶颈**：multi 需核验 4 个选项，窄证据漏选严重；但过宽证据又致过选。

**架构**（`agent/kg_reasoner.py`）：
- **Round 1 构图（0 token）**：claim 节点（每选项原子断言，复用 `ImportanceScorer.decompose_option_claims`）、evidence 节点（题干关键词全局定位段 + 每选项专属 grep 段）、fact 节点（数字/年份）；边为规则判定——hard（数字+单位逐字命中或实体关键词逐字）、soft（同义命中）、矛盾（留 Qwen）。
- **Round 2 V35 基线**：广域 60K 证据（head 在前 + 锚词追加）+ V20 prompt + Qwen 出初判 draft。
- **Round 3 add-only 图审计**：对 draft 未选选项，用图专属证据 + 证据门控（硬须数字+单位逐字，软须明确同义）判定是否补充。**仅增不删**（`set(ans) ⊇ set(draft)`），保证不回归。
- **后处理不截断**：允许 ABCD 4 字母（V20 旧守门截到 3 字母会砍真值 ABCD 题）。

**创新点**：图 = 压缩记忆（节点+边 ≪ 原文），多轮图遍历 = 逐选项断言沿边验证；规则构图零 token，最终答案来自 Qwen 当题推理，合规。

### 2.2 tf/mcq（35题）：靶向证据覆盖修复 + 宽容 prompt（V43）

**瓶颈**（小样本验证定位）：tf/mcq 走 V30 窄结构化证据（8–14K），截掉深位事实（如 43.24%@文档深位 51168、2019 回购@深位 54159、宁德时代市占率），模型看不到证据→判错。**真瓶颈是证据覆盖，非 prompt 宽容度**（宽容 alone 仅 +3/10）。

**架构**（`agent/reasoner_v43.py`）：
- **V30 结构化基础 + 靶向 grep**：从陈述（tf）或所有选项（mcq）抽取关键数+实体，grep 原文补深位段。
- **无分词器关键词提取**：文档锚定最长 CJK 匹配——每位置取原文真实存在的最长子串，自动发现"力诺投资/宁德时代/回售"等实体，不产碎片；数字须带单位/小数/%（滤 doc-id 001/004）。
- **频率感知 grep**：稀有词（低频，如 150%）先取 snippet，泛词（高频，如 2024）后取，轮询每词 ≥1 条，防泛词占满预算。
- **子串去重**：A 是 B 子串则去 A，消除冗余碎片（期债券发行/债券发行/券发行）腾 slot。
- **宽容 PROMPT_TF/MCQ**：仿审计 soft-tolerance，容忍同义/改述/概括，仅否决硬矛盾（数值/主体/方向/年份/时限/单位）。

## 3. 实验演进

| 版本 | Acc(对/100) | Token | Score | 关键改变 |
|------|------|------|------|------|
| V31 | 60 | 3.19M | 48.50 | multi V22 完整 + tf/mcq V30 精炼（动态证据分配）|
| V42 | 73 | 3.23M | 58.87 | + add-only 图审计修 multi 漏选（+8 multi）|
| **V43** | **~85** | **~3.42M** | **~67.6** | + tf/mcq 靶向证据+宽容（tf/mcq 21→31 一致）|

**关键教训**：
- 压缩与正确率 1:1 交换（V40/V42-v6 三次证实）：窄 grep 丢证据，审计窄 grep 无法补回广域证据丢失。故 multi 保 60K 基线。
- 过严是隐形漏选：V35 `PROMPT_TF`"用词替换→判 B"过严，命题人按语义判。宽容+靶向证据修 11 道 tf/mcq 翻转。
- test_A（92 对）比 JSON 真值可靠（5 题源文裁决 4:1）；V43 在 res_a_006 上甚至比 test_A 更对（识别"除客户资金杠杆"漏字）。

## 4. 合规说明

- 推理全程仅调用 Qwen API（`qwen_client.py`，OpenAI 兼容接口，`qwen3.6-plus`，temperature=0.1）。
- 图构图、靶向 grep、同义边均为**规则**判定，零 token，非"模型"；不构成预处理语义摘要答题。
- 最终答案来自 Qwen 当题推理；无 rerank/投票/纠错模型，无非 Qwen 模型参与。
- 不微调基座模型。

## 5. 运行

### 环境准备
```bash
pip install -r requirements.txt
# .env 配置 DASHSCOPE_API_KEY
```

### 预处理（首次）
```bash
python -c "from agent.pdf_parser import preprocess_all; preprocess_all()"   # PDF→文本
python -c "from agent.chunker import rebuild_structured_index; rebuild_structured_index()"  # 索引
```

### A 榜全量推理
```bash
python -m agent.pipeline_v43
```
输出：`results/answer_v43.csv`（qid,answer）+ `results/eval_results_v42.json`（multi cot）。

### 证据导出（零 token，复用证据构造）
```bash
python _gen_evidence.py
```
输出：`results/evidence.json`（每题证据片段+答案+策略）。

## 6. 代码结构（V43 依赖链）

```
agent/
├── pipeline_v43.py        # 全量 runner（A 榜入口）
├── reasoner_v43.py        # V43: tf/mcq 靶向证据+宽容; multi 委托 V42
├── kg_reasoner.py         # V42: GraphBuilder + GraphTraverser + ReasoningAgentV42 (multi)
├── reasoner_v35.py        # V35: multi 广域60K证据 (head+锚词); build_evidence_v35
├── reasoner_v31.py / v30.py / v22.py / v21.py / v20.py  # multi/tf 证据与 prompt 基线链
├── structured_evidence.py # V30 域专属结构化提取器
├── indexer.py             # DocumentIndex 文档加载
├── memory_compressor_v41.py  # ImportanceScorer (claim 分解)
├── reasoner_multiround.py # extract_keywords / locate_sections
├── synonym_expander.py    # SYNONYM_MAP (同义边)
├── anchor_state_v41.py    # AnchorState (锚定状态)
├── qwen_client.py         # Qwen API 封装
├── postprocessor.py       # 答案抽取 + CSV 生成
├── config.py / chunker.py / pdf_parser.py
```
注：`agent/` 内 v6–v34 等历史版本为实验记录，V43 实际依赖见上表（`pipeline_v43` → `reasoner_v43` → `kg_reasoner`/`reasoner_v35` → v31/v30/v22/v20）。

## 7. 结果

- **A 榜 V43**：~85 对 / ~3.42M token → **预估 Score ≈ 67.6**（V31=48.50, V42=58.87 基础上 +9.1）。
- tf/mcq 一致 test_A 21→31（+10 净，11 修对 1 回归），multi 54/65（图审计 +8）。
- 合规（仅 Qwen），代码审核包：`answer.csv` + `evidence.json` + `agent/` + `script/` + `logs/` + `processed_data/` + `requirements.txt` + `README.md`。
