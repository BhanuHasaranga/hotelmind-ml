"""Exercises the concrete provider SDK call paths (generate/stream) using
mocked SDK clients injected via monkeypatching the provider's `_client`
method -- no real network calls are made.
"""

from __future__ import annotations

import sys
import types

import pytest

from genai.llm.gemini_provider import GeminiProvider
from genai.llm.ollama_provider import OllamaProvider
from genai.llm.openai_provider import OpenAIProvider


# --- OpenAI ---


class _FakeOpenAIMessage:
    content = "hello from openai"


class _FakeOpenAIChoice:
    message = _FakeOpenAIMessage()


class _FakeOpenAIUsage:
    prompt_tokens = 5
    completion_tokens = 7


class _FakeOpenAIResponse:
    choices = [_FakeOpenAIChoice()]
    usage = _FakeOpenAIUsage()


class _FakeOpenAIStreamChunkDelta:
    def __init__(self, content):
        self.content = content


class _FakeOpenAIStreamChoice:
    def __init__(self, content):
        self.delta = _FakeOpenAIStreamChunkDelta(content)


class _FakeOpenAIStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeOpenAIStreamChoice(content)]


class _FakeOpenAIChatCompletions:
    def create(self, model, messages, temperature, max_tokens, stream=False):
        if stream:
            return iter([_FakeOpenAIStreamChunk("hel"), _FakeOpenAIStreamChunk("lo")])
        return _FakeOpenAIResponse()


class _FakeOpenAIChat:
    completions = _FakeOpenAIChatCompletions()


class _FakeOpenAIClient:
    chat = _FakeOpenAIChat()


def test_openai_provider_generate(monkeypatch):
    provider = OpenAIProvider(model="gpt-4o-mini")
    provider._api_key = "fake-key"
    monkeypatch.setattr(provider, "_client", lambda: _FakeOpenAIClient())
    result = provider.generate("hi", system="be nice")
    assert result.text == "hello from openai"
    assert result.prompt_tokens == 5
    assert result.provider == "openai"


def test_openai_provider_stream(monkeypatch):
    provider = OpenAIProvider(model="gpt-4o-mini")
    provider._api_key = "fake-key"
    monkeypatch.setattr(provider, "_client", lambda: _FakeOpenAIClient())
    chunks = list(provider.stream("hi"))
    assert chunks == ["hel", "lo"]


def test_openai_provider_client_requires_key():
    provider = OpenAIProvider(model="gpt-4o-mini")
    provider._api_key = None
    with pytest.raises(RuntimeError):
        provider._client()


# --- Gemini ---


class _FakeGeminiUsage:
    prompt_token_count = 4
    candidates_token_count = 6


class _FakeGeminiResponse:
    text = "hello from gemini"
    usage_metadata = _FakeGeminiUsage()


class _FakeGeminiStreamChunk:
    def __init__(self, text):
        self.text = text


class _FakeGeminiModels:
    def generate_content(self, model, contents, config):
        return _FakeGeminiResponse()

    def generate_content_stream(self, model, contents, config):
        return iter([_FakeGeminiStreamChunk("g1"), _FakeGeminiStreamChunk("g2")])


class _FakeGeminiClient:
    models = _FakeGeminiModels()


def test_gemini_provider_generate(monkeypatch):
    provider = GeminiProvider(model="gemini-1.5-flash")
    provider._api_key = "fake-key"
    monkeypatch.setattr(provider, "_client", lambda: _FakeGeminiClient())
    monkeypatch.setattr(provider, "_config", lambda system=None, **kwargs: object())
    result = provider.generate("hi", system="be nice")
    assert result.text == "hello from gemini"
    assert result.prompt_tokens == 4


def test_gemini_provider_stream(monkeypatch):
    provider = GeminiProvider(model="gemini-1.5-flash")
    provider._api_key = "fake-key"
    monkeypatch.setattr(provider, "_client", lambda: _FakeGeminiClient())
    monkeypatch.setattr(provider, "_config", lambda system=None, **kwargs: object())
    chunks = list(provider.stream("hi"))
    assert chunks == ["g1", "g2"]


def test_gemini_provider_client_requires_key():
    provider = GeminiProvider(model="gemini-1.5-flash")
    provider._api_key = None
    with pytest.raises(RuntimeError):
        provider._client()


# --- Ollama ---


class _FakeOllamaClient:
    def chat(self, model, messages, options, stream=False):
        if stream:
            return iter([{"message": {"content": "o1"}}, {"message": {"content": "o2"}}])
        return {"message": {"content": "hello from ollama"}, "prompt_eval_count": 3, "eval_count": 9}


def test_ollama_provider_generate(monkeypatch):
    provider = OllamaProvider(model="llama3.1")
    monkeypatch.setattr(provider, "_client", lambda: _FakeOllamaClient())
    result = provider.generate("hi", system="be nice")
    assert result.text == "hello from ollama"
    assert result.completion_tokens == 9


def test_ollama_provider_stream(monkeypatch):
    provider = OllamaProvider(model="llama3.1")
    monkeypatch.setattr(provider, "_client", lambda: _FakeOllamaClient())
    chunks = list(provider.stream("hi"))
    assert chunks == ["o1", "o2"]


def test_ollama_provider_is_available_reachable(monkeypatch):
    provider = OllamaProvider(model="llama3.1")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=1: _FakeResponse())
    assert provider.is_available() is True
