"""Selects a concrete `LLMProvider` purely from `genai_settings.LLM_PROVIDER`.

Callers (routers, RAG chain, insights service) depend only on
`genai.llm.base.LLMProvider` and call `get_llm_provider()` — never import a
concrete provider class directly — so switching providers is a config-only
change with zero code edits.
"""

from genai.config.genai_settings import GenAISettings, genai_settings
from genai.llm.base import LLMProvider
from genai.llm.gemini_provider import GeminiProvider
from genai.llm.ollama_provider import OllamaProvider
from genai.llm.openai_provider import OpenAIProvider
from genai.observability.llm_logging import LoggingLLMProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def get_llm_provider(settings: GenAISettings | None = None, *, with_logging: bool = True) -> LLMProvider:
    settings = settings or genai_settings
    provider_key = settings.LLM_PROVIDER.lower()
    provider_cls = _PROVIDERS.get(provider_key)
    if provider_cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider_key!r}; expected one of {sorted(_PROVIDERS)}"
        )
    logger.info("Instantiating LLM provider=%s model=%s", provider_key, settings.LLM_MODEL)
    provider = provider_cls(
        model=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )
    return LoggingLLMProvider(provider) if with_logging else provider
