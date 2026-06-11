"""AFAC2026 赛题四：金融长文本Agent 配置文件"""
import os
from dotenv import load_dotenv

load_dotenv()

# API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置
MODEL_NAME = "qwen-plus"  # 评测基准模型，可切换为 qwen-max / qwen-turbo
MODEL_MAX_TOKENS = 131072  # Qwen-plus 最大上下文

# Token 预算
TOKEN_BUDGET = 5_000_000

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public_dataset_upload")
RAW_DIR = os.path.join(DATA_DIR, "raw")
QUESTIONS_DIR = os.path.join(DATA_DIR, "questions")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# 领域配置
DOMAINS = ["insurance", "regulatory", "financial_contracts", "financial_reports", "research"]

# 检索配置
CHUNK_SIZE = 2000      # 每个文本块字符数
CHUNK_OVERLAP = 200    # 块重叠字符数
TOP_K_CHUNKS = 5       # 每题检索的 top-k 文本块数

# 推理配置
MAX_RETRIES = 3        # API 调用最大重试次数
TEMPERATURE = 0.1      # 低温度保证稳定输出
