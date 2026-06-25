"""Qwen API 调用封装

通过阿里云百炼平台调用 Qwen 系列模型 API。
使用 OpenAI 兼容接口格式。
"""
import time
import json
from openai import OpenAI
from agent.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME, MAX_RETRIES


class QwenClient:
    """Qwen 模型 API 客户端"""

    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
        # Token 统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 4096,
             timeout: float = 120, enable_thinking: bool = True) -> dict:
        """
        调用 Qwen Chat API

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            timeout: 超时秒数（默认120秒）
            enable_thinking: qwen3 系列是否启用思考模式（默认 True）
        """
        extra_body = {}
        if not enable_thinking:
            extra_body["enable_thinking"] = False

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    extra_body=extra_body or None,
                )
                
                content = response.choices[0].message.content
                usage = response.usage
                
                result = {
                    "content": content,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }
                
                # 累计统计
                self.total_prompt_tokens += result["prompt_tokens"]
                self.total_completion_tokens += result["completion_tokens"]
                self.total_tokens += result["total_tokens"]
                self.call_count += 1
                
                return result
                
            except Exception as e:
                print(f"  API 调用失败 (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def get_token_stats(self) -> dict:
        """获取 Token 使用统计"""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def reset_stats(self):
        """重置 Token 统计"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
