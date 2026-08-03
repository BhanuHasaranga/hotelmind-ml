"""Provider-agnostic LLM abstraction.

Every concrete provider (OpenAI, Gemini, Ollama) implements this ABC so that
callers (routers, RAG chain, insights service) never depend on a concrete
SDK. Switching `LLM_PROVIDER` in config is a config-only change.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class LLMResult:
    """Return value of a non-streaming `generate()` call."""

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for chat/completion-style LLM providers."""

    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResult:
        """Return a complete generation as an `LLMResult`."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        """Yield text chunks as they are generated."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this provider is minimally configured to be called live.

        Subclasses override to check API keys / reachability. Used by
        higher layers (insights, RAG chain) to gracefully degrade to
        rule-based/template output instead of hard-failing.
        """
        return True
