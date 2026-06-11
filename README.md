# AFAC2026 赛题四：金融长文本Agent的动态记忆压缩与高效问答挑战

## 赛题概述

- **赛题名称**：金融长文本Agent的动态记忆压缩与高效问答挑战
- **赛题编号**：AFAC2026 挑战组赛题四
- **平台**：阿里云天池
- **评测公式**：`FinalScore = 100 × Accuracy × (0.7 + 0.3 × TokenScore)`
- **Token 预算**：5,000,000
- **模型限制**：仅 Qwen 系列 API（阿里云百炼/魔搭），不微调

## 项目结构

```
AFAC_4/
├── .env                          # API Key 配置
├── .gitignore                    # Git 忽略规则
├── requirements.txt              # Python 依赖
├── competition_info.md           # 赛事介绍
├── task_details.md               # 赛题与数据详情
├── public_dataset_upload/        # 原始数据（不提交）
│   ├── raw/                      # 86+ 个原始 PDF 文档
│   │   ├── insurance/            # 16 个保险条款 PDF
│   │   ├── regulatory/           # 26 个监管法规（PDF+HTML+TXT）
│   │   ├── financial_contracts/ # 14 个金融合同 PDF
│   │   ├── financial_reports/    # 10 个年报 PDF
│   │   └── research/            # 20 个研报 PDF
│   └── questions/
│       └── group_a/              # A 榜题目 JSON
├── processed_data/               # 预处理后数据（不提交）
│   ├── insurance/                # 保险条款结构化 JSON
│   ├── regulatory/               # 监管法规结构化 JSON
│   ├── financial_contracts/       # 金融合同结构化 JSON
│   ├── financial_reports/        # 年报结构化 JSON
│   ├── research/                 # 研报结构化 JSON
│   └── structured_index.json    # 结构化分块索引
├── results/                      # 输出结果（不提交）
│   └── answer.csv
└── agent/                        # 核心代码
    ├── __init__.py
    ├── config.py                 # 全局配置
    ├── qwen_client.py            # Qwen API 封装
    ├── pdf_parser.py             # PDF 文档预处理器
    ├── chunker.py                # 结构化分块器
    ├── indexer.py                 # BM25 索引 + Qwen Rerank
    ├── clause_locator.py         # 条款精准定位
    ├── reasoner.py                # 推理引擎 V3
    ├── postprocessor.py           # 答案后处理与 CSV 生成
    └── pipeline.py                # 主流水线
```

## 技术架构

### 整体流程

```
原始 PDF
  ↓ pdf_parser.py (PyMuPDF + pdfplumber 双引擎)
预处理 JSON
  ↓ chunker.py (按条款/章节/表格结构化分块)
结构化索引 (4881 块)
  ↓ indexer.py (BM25 粗召回 + Qwen Rerank 精排)
  ↓ clause_locator.py (条款编号精准定位)
精选证据
  ↓ reasoner.py (领域专用 Prompt + 两阶段推理 + 自校验)
  ↓ postprocessor.py (答案校验 + 格式标准化)
answer.csv
```

### 核心技术点

#### 1. 文档预处理（`pdf_parser.py`）
- **双引擎解析**：PyMuPDF（通用文本）+ pdfplumber（表格提取），合并取各自优势
- 保留页码、标题、章节结构
- 支持 PDF / HTML / TXT 三种来源

#### 2. 结构化分块（`chunker.py`）
- **不再使用固定窗口**，而是按语义边界分块：
  - 监管法规：按「第X条」条款切分，每块 1-3 条
  - 保险条款：按条款+计算公式切分
  - 财务报表：按章节切分，标记含财务数据的块
  - 金融合同：按条款切分
  - 研报：按章节切分
- 结果：573 个文档 → 4881 个语义完整的块

#### 3. 智能检索（`indexer.py`）
- **BM25 粗召回**：jieba 分词 + BM25Okapi
- **Qwen Rerank 精排**：用 Qwen 判断候选块与问题的相关性，精选最相关证据
- **多查询策略**：题干 + 各选项分别检索，合并去重
- **精准文档定位**：A 榜已知 doc_ids 时，限定检索范围

#### 4. 条款精准定位（`clause_locator.py`）
- 从题目和选项中提取条款引用（如"第47条"、"第82条"）
- 直接在文档中定位该条款原文
- 中文数字→阿拉伯数字转换（"四十七" → 47）
- 精准定位的条款放入证据最前面

#### 5. 推理引擎 V3（`reasoner.py`）
- **5 领域专用系统提示**：保险精算师、监管合规专家、合同分析师、财务分析师、研报专家
- **选项逐项验证**：每个选项独立判断正确/错误，输出 `选项X：[正确/错误] — 依据：...`
- **两阶段推理**（多选题）：
  - Stage 1：提取关键事实（数字、条款、条件）
  - Stage 2：基于事实判断各选项
- **自校验机制**：对多选题答案做二次确认，宁可少选不多选

#### 6. 答案后处理（`postprocessor.py`）
- 多策略答案提取（最终答案→答案标记→独立行→选项引用→全文扫描）
- 格式标准化：单选/判断取首字母，多选去重排序
- CSV 生成含 Token 统计

#### 7. Token 预算管控（`pipeline.py`）
- 5M 预算分级分配：Rerank 20% + 推理 80%
- 简单题优先（tf > mcq > multi）
- 超 95% 预算自动停止

## 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 配置 API Key

创建 `.env` 文件：
```
DASHSCOPE_API_KEY=sk-your-api-key-here
```

### 运行预处理

```bash
# PDF 预处理（首次运行，约 5-10 分钟）
python -c "from agent.pdf_parser import preprocess_all; preprocess_all()"

# 构建结构化索引
python -c "from agent.chunker import rebuild_structured_index; rebuild_structured_index()"
```

### 运行 A 榜评测

```bash
python -m agent.pipeline
```

输出文件：`results/answer.csv`

## 评测指标

| 指标 | 公式 |
|------|------|
| 准确率 | Accuracy = Correct / Total |
| Token 效率 | TokenScore = max(0, min(1, (5,000,000 - TotalTokens) / 5,000,000)) |
| 最终得分 | FinalScore = 100 × Accuracy × (0.7 + 0.3 × TokenScore) |

## Baseline 对比

| 组别 | Baseline 准确率 | Baseline Token |
|------|----------------|----------------|
| A 组 | 17.0% | 3,628,186 |
| B 组 | 13.0% | 3,884,045 |
| 总计 | 15.0% | 7,512,231 |

### V3 优化预期

| 优化项 | 预期效果 |
|--------|---------|
| 结构化分块 | Accuracy +10-20% (保持条款完整性) |
| 领域专用 Prompt | Accuracy +10-15% (专业推理) |
| 条款精准定位 | Accuracy +5-10% (监管/合同领域) |
| 选项逐项验证 | Accuracy +10% (多选题关键) |
| 两阶段推理 | Accuracy +5% + Token 效率提升 |
| 自校验 | Accuracy +3-5% (减少错选) |
| Qwen Rerank | Accuracy +5% + Token 效率提升 |

## 依赖

- `openai` - Qwen API (OpenAI 兼容接口)
- `python-dotenv` - 环境变量
- `pymupdf` / `pdfplumber` - PDF 解析
- `jieba` - 中文分词
- `rank_bm25` - BM25 检索
- `pandas` - CSV 生成
- `beautifulsoup4` - HTML 解析

## 注意事项

- `processed_data/` 和 `results/` 目录不提交到 Git（太大）
- 预处理数据本地生成，运行前需先执行预处理步骤
- API Key 存放在 `.env` 文件中，不提交到 Git
- 推理问答阶段仅使用 Qwen 系列 API，预处理阶段可用其他工具
