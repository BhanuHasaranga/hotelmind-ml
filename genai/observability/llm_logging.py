"""Token usage + latency logging wrapper around LLM provider calls.

Wraps any `LLMProvider` instance so every `generate()`/`stream()` call is
logged via the existing `src.utils.logging.get_logger` pattern: provider,
model, tokens in/out, latency ms. Use `LoggingLLMProvider(provider)` to wrap
any concrete provider, or `log_call(...)` directly around a raw `LLMResult`.
"""

import time
from collections.abc import Iterator

from genai.llm.base import LLMProvider, LLMResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def log_result(result: LLMResult) -> None:
    logger.info(
        "llm_call provider=%s model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%.1f",
        result.provider,
        result.model,
        result.prompt_tokens,
        result.completion_tokens,
        result.latency_ms,
    )


class LoggingLLMProvider(LLMProvider):
    """Decorator that wraps any `LLMProvider` and logs every call."""

    def __init__(self, inner: LLMProvider) -> None:
        super().__init__(inner.model, inner.temperature, inner.max_tokens)
        self._inner = inner
        self.name = inner.name

    def is_available(self) -> bool:
        return self._inner.is_available()

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResult:
        result = self._inner.generate(prompt, system=system, **kwargs)
        log_result(result)
        return result

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        start = time.perf_counter()
        chunk_count = 0
        char_count = 0
        for chunk in self._inner.stream(prompt, system=system, **kwargs):
            chunk_count += 1
            char_count += len(chunk)
            yield chunk
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "llm_stream provider=%s model=%s chunks=%d chars=%d latency_ms=%.1f",
            self._inner.name,
            self._inner.model,
            chunk_count,
            char_count,
            latency_ms,
        )
