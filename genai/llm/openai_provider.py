"""OpenAI chat-completions provider implementation."""

import time
from collections.abc import Iterator

from genai.config.genai_settings import genai_settings
from genai.llm.base import LLMProvider, LLMResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024) -> None:
        super().__init__(model, temperature, max_tokens)
        self._api_key = genai_settings.OPENAI_API_KEY

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client(self):
        from openai import OpenAI

        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return OpenAI(api_key=self._api_key)

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResult:
        client = self._client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
        )

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        client = self._client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
