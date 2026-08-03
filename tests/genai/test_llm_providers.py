import pytest

from genai.config.genai_settings import GenAISettings
from genai.llm.base import LLMProvider, LLMResult
from genai.llm.factory import get_llm_provider
from genai.llm.gemini_provider import GeminiProvider
from genai.llm.ollama_provider import OllamaProvider
from genai.llm.openai_provider import OpenAIProvider
from genai.observability.llm_logging import LoggingLLMProvider


class FakeProvider(LLMProvider):
    name = "fake"

    def generate(self, prompt, system=None, **kwargs):
        return LLMResult(text=f"echo:{prompt}", provider=self.name, model=self.model, prompt_tokens=3, completion_tokens=4, latency_ms=1.0)

    def stream(self, prompt, system=None, **kwargs):
        for chunk in ["a", "b", "c"]:
            yield chunk


def test_factory_selects_openai():
    settings = GenAISettings(LLM_PROVIDER="openai", LLM_MODEL="gpt-4o-mini")
    provider = get_llm_provider(settings, with_logging=False)
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"


def test_factory_selects_gemini():
    settings = GenAISettings(LLM_PROVIDER="gemini", LLM_MODEL="gemini-1.5-flash")
    provider = get_llm_provider(settings, with_logging=False)
    assert isinstance(provider, GeminiProvider)


def test_factory_selects_ollama():
    settings = GenAISettings(LLM_PROVIDER="ollama", LLM_MODEL="llama3.1")
    provider = get_llm_provider(settings, with_logging=False)
    assert isinstance(provider, OllamaProvider)


def test_factory_wraps_with_logging_by_default():
    settings = GenAISettings(LLM_PROVIDER="ollama")
    provider = get_llm_provider(settings)
    assert isinstance(provider, LoggingLLMProvider)


def test_factory_unknown_provider_raises():
    settings = GenAISettings(LLM_PROVIDER="bogus")
    with pytest.raises(ValueError):
        get_llm_provider(settings)


def test_openai_provider_not_available_without_key():
    provider = OpenAIProvider(model="gpt-4o-mini")
    provider._api_key = None
    assert provider.is_available() is False


def test_gemini_provider_not_available_without_key():
    provider = GeminiProvider(model="gemini-1.5-flash")
    provider._api_key = None
    assert provider.is_available() is False


def test_ollama_provider_unreachable_returns_false():
    provider = OllamaProvider(model="llama3.1")
    provider._base_url = "http://localhost:1"  # unlikely to be listening
    assert provider.is_available() is False


def test_logging_wrapper_generate_passthrough():
    fake = FakeProvider(model="fake-model")
    wrapped = LoggingLLMProvider(fake)
    result = wrapped.generate("hello")
    assert result.text == "echo:hello"
    assert wrapped.is_available() is True


def test_logging_wrapper_stream_passthrough():
    fake = FakeProvider(model="fake-model")
    wrapped = LoggingLLMProvider(fake)
    chunks = list(wrapped.stream("hi"))
    assert chunks == ["a", "b", "c"]
