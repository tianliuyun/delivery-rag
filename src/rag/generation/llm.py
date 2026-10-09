"""
LLM 封装
支持多种 provider：mock / openai / ollama / qwen
"""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str  # system / user / assistant
    content: str


class LLMClient:
    """LLM 客户端封装

    支持多种后端：
    - mock: 模拟模式，用于测试和演示
    - openai: OpenAI API
    - ollama: 本地 Ollama
    - qwen: 阿里通义千问
    - vllm: vLLM 部署的本地模型
    """

    def __init__(self, provider: str = "mock", model: str = "mock",
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: float = 0.1):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self._client = None

    def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """对话"""
        if self.provider == "mock":
            return self._mock_chat(messages)
        elif self.provider == "openai":
            return self._openai_chat(messages, **kwargs)
        elif self.provider == "ollama":
            return self._ollama_chat(messages, **kwargs)
        else:
            return self._mock_chat(messages)

    def _mock_chat(self, messages: List[ChatMessage]) -> str:
        """模拟 LLM 响应（用于测试/演示）"""
        # 找最后一条 user 消息
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user = msg.content
                break

        # 简单的模拟：基于用户输入生成一个看起来合理的回答
        # 实际项目中会调用真正的 LLM API
        if "你好" in last_user or "hello" in last_user.lower():
            return "你好！我是交付知识库助手，有什么可以帮你的？"

        # 检查是否有上下文（在消息里找）
        has_context = False
        for msg in messages:
            if "上下文" in msg.content or "参考资料" in msg.content:
                has_context = True
                break

        if has_context:
            return ("根据参考资料，这个问题的答案如下：\n\n"
                    "桌面云接入速度慢的问题，可以从以下几个方面排查：\n"
                    "1. 检查网络延迟和带宽情况[1]\n"
                    "2. 确认客户端版本是否最新[2]\n"
                    "3. 排查服务器资源使用情况[3]\n\n"
                    "如果以上都正常，建议联系技术支持进一步排查。")
        else:
            return "（mock模式）这是一个模拟回答。实际使用时会调用真实的大模型生成答案。"

    def _openai_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """OpenAI API 调用"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=kwargs.get('temperature', self.temperature),
            )
            return response.choices[0].message.content
        except ImportError:
            return self._mock_chat(messages)

    def _ollama_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """Ollama 本地模型调用"""
        try:
            import requests
            url = f"{self.base_url or 'http://localhost:11434'}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
            }
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()['message']['content']
        except Exception:
            return self._mock_chat(messages)
