"""Google Gemini provider implementation (via `google-genai` SDK)."""

import time
from collections.abc import Iterator

from genai.config.genai_settings import genai_settings
from genai.llm.base import LLMProvider, LLMResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024) -> None:
        super().__init__(model, temperature, max_tokens)
        self._api_key = genai_settings.GEMINI_API_KEY

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client(self):
        from google import genai as google_genai

        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        return google_genai.Client(api_key=self._api_key)

    def _config(self, **kwargs):
        from google.genai import types

        return types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            system_instruction=kwargs.get("system"),
        )

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResult:
        client = self._client()
        start = time.perf_counter()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._config(system=system, **kwargs),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            latency_ms=latency_ms,
        )

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        client = self._client()
        stream = client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=self._config(system=system, **kwargs),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
