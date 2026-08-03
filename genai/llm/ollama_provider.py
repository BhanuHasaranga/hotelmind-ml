"""Local Ollama provider implementation.

No API key required; only requires a reachable Ollama server
(`OLLAMA_BASE_URL`, default `http://localhost:11434`).
"""

import time
from collections.abc import Iterator

from genai.config.genai_settings import genai_settings
from genai.llm.base import LLMProvider, LLMResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024) -> None:
        super().__init__(model, temperature, max_tokens)
        self._base_url = genai_settings.OLLAMA_BASE_URL

    def is_available(self) -> bool:
        import urllib.error
        import urllib.request

        try:
            urllib.request.urlopen(self._base_url, timeout=1)
            return True
        except urllib.error.URLError:
            return False
        except Exception:
            return False

    def _client(self):
        import ollama

        return ollama.Client(host=self._base_url)

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResult:
        client = self._client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response["message"]["content"]
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            latency_ms=latency_ms,
        )

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        client = self._client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for chunk in client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        ):
            content = chunk.get("message", {}).get("content")
            if content:
                yield content
