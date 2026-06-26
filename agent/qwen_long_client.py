"""Qwen-Long 文件上传客户端

核心优势（相比 qwen-plus + 分块检索）：
- 1000万 Token 上下文，可直接喂入完整 PDF 无需分块
- 价格 0.0005元/千Token（比 qwen-plus 便宜约10倍）
- 文件上传/存储/解析免费
- 支持 PDF 直接上传，服务端解析（不依赖本地 pymupdf/pdfplumber）
- 单次最多引用100个文件

使用流程：
  1. upload_file(pdf_path) → file_id（首次上传）
  2. chat_with_files(file_ids, messages) → 回答（qwen-long 处理）
  3. file_id 缓存到磁盘，同一文件不重复上传

注意：
  - 每次调用时文件 Token 都会计入本次 input token
  - 含 fileid system message 时，user content ≤ 9000 tokens
  - Token 计入比赛总预算（qwen-long tokens 也算）
"""
import os
import json
import time
import hashlib
from pathlib import Path
from openai import OpenAI
from agent.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

# 文件 ID 缓存路径（避免重复上传）
FILE_ID_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cache", "file_ids.json"
)

QWEN_LONG_MODEL = "qwen-long"


def _file_hash(path: str) -> str:
    """用文件 MD5 作为缓存 key"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class QwenLongClient:
    """Qwen-Long 文件上传与问答客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
        self._file_id_cache: dict = {}   # md5 → file_id
        self._doc_id_cache: dict = {}    # doc_id → file_id
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.upload_count = 0
        self._load_cache()

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def _load_cache(self):
        os.makedirs(os.path.dirname(FILE_ID_CACHE), exist_ok=True)
        if os.path.exists(FILE_ID_CACHE):
            with open(FILE_ID_CACHE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._file_id_cache = data.get("md5", {})
            self._doc_id_cache = data.get("doc_id", {})

    def _save_cache(self):
        with open(FILE_ID_CACHE, "w", encoding="utf-8") as f:
            json.dump({
                "md5": self._file_id_cache,
                "doc_id": self._doc_id_cache,
            }, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 文件上传
    # ------------------------------------------------------------------

    def upload_file(self, file_path: str, doc_id: str = None) -> str:
        """上传文件，返回 file_id（带缓存）"""
        # 先查 doc_id 缓存
        if doc_id and doc_id in self._doc_id_cache:
            return self._doc_id_cache[doc_id]

        # 再查 md5 缓存
        file_md5 = _file_hash(file_path)
        if file_md5 in self._file_id_cache:
            fid = self._file_id_cache[file_md5]
            if doc_id:
                self._doc_id_cache[doc_id] = fid
            return fid

        # 上传到 DashScope
        for attempt in range(3):
            try:
                with open(file_path, "rb") as fp:
                    file_object = self.client.files.create(
                        file=(Path(file_path).name, fp),
                        purpose="file-extract",
                    )
                fid = file_object.id
                self._file_id_cache[file_md5] = fid
                if doc_id:
                    self._doc_id_cache[doc_id] = fid
                self._save_cache()
                self.upload_count += 1
                print(f"    上传: {Path(file_path).name} → {fid}")
                return fid
            except Exception as e:
                print(f"    上传失败 (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
        return ""

    def get_file_id(self, doc_id: str) -> str:
        """从缓存获取 file_id"""
        return self._doc_id_cache.get(doc_id, "")

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------

    def chat_with_files(
        self,
        file_ids: list,
        question: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 300,
    ) -> dict:
        """用 qwen-long 对上传的文件进行问答

        Args:
            file_ids: 文件 ID 列表（最多100个）
            question: 用户问题（含 prompt 格式）
            system_prompt: 领域系统提示
            temperature: 温度
            max_tokens: 最大输出
            timeout: 超时秒数（长文档需更长）
        """
        if not file_ids:
            return {"content": "", "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0}

        # 构建 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # fileid 引用：多文件逗号分隔
        fileid_str = ",".join(f"fileid://{fid}" for fid in file_ids if fid)
        messages.append({"role": "system", "content": fileid_str})

        # user 问题（含 fileid 时限 ≤9000 tokens，约36000 chars）
        user_content = question[:35000]  # 安全截断
        messages.append({"role": "user", "content": user_content})

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=QWEN_LONG_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                content = resp.choices[0].message.content
                usage = resp.usage
                result = {
                    "content": content,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }
                self.total_prompt_tokens += result["prompt_tokens"]
                self.total_completion_tokens += result["completion_tokens"]
                self.total_tokens += result["total_tokens"]
                self.call_count += 1
                return result
            except Exception as e:
                print(f"  qwen-long 调用失败 (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))

        return {"content": "", "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0}

    def get_token_stats(self) -> dict:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "upload_count": self.upload_count,
        }
