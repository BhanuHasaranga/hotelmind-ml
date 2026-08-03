"""Exercises the generate()/stream() implementations of each concrete LLM
provider against mocked SDK clients (openai/google-genai/ollama) -- no real
network calls, no API keys required."""

from __future__ import annotations

from types import SimpleNamespace

from genai.llm.gemini_provider import GeminiProvider
from genai.llm.ollama_provider import OllamaProvider
from genai.llm.openai_provider import OpenAIProvider


# --- OpenAI ---


class _FakeOpenAIMessage:
    def __init__(self, content):
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content):
        self.message = _FakeOpenAIMessage(content)
        self.delta = SimpleNamespace(content=content)


class _FakeOpenAIResponse:
    def __init__(self, content, prompt_tokens=10, completion_tokens=5):
        self.choices = [_FakeOpenAIChoice(content)]
        self.usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class _FakeOpenAIClient:
    def __init__(self, response_text="hello from openai"):
        self._text = response_text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature, max_tokens, stream=False):
        return _FakeOpenAIResponse(self._text)


class _FakeOpenAIChoiceStream:
    def __init__(self, content):
        self.delta = SimpleNamespace(content=content)


class _FakeOpenAIStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeOpenAIChoiceStream(content)]


def test_openai_generate_returns_result(monkeypatch):
    provider = OpenAIProvider(model="gpt-4o-mini")
    provider._api_key = "sk-test"
    fake_client = _FakeOpenAIClient(response_text="Hello, guest!")
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    result = provider.generate("hi", system="be nice")
    assert result.text == "Hello, guest!"
    assert result.provider == "openai"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def test_openai_stream_yields_chunks(monkeypatch):
    provider = OpenAIProvider(model="gpt-4o-mini")
    provider._api_key = "sk-test"

    class _StreamClient:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda model, messages, temperature, max_tokens, stream: [
                    _FakeOpenAIStreamChunk("Hel"),
                    _FakeOpenAIStreamChunk("lo"),
                ]
            )
        )

    monkeypatch.setattr(provider, "_client", lambda: _StreamClient())
    chunks = list(provider.stream("hi"))
    assert chunks == ["Hel", "lo"]


# --- Gemini ---


class _FakeGeminiResponse:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = SimpleNamespace(prompt_token_count=8, candidates_token_count=4)


class _FakeGeminiChunk:
    def __init__(self, text):
        self.text = text


def test_gemini_generate_returns_result(monkeypatch):
    provider = GeminiProvider(model="gemini-1.5-flash")
    provider._api_key = "test-key"

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda model, contents, config: _FakeGeminiResponse("Hi from Gemini"))
    )
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    result = provider.generate("hi")
    assert result.text == "Hi from Gemini"
    assert result.provider == "gemini"
    assert result.prompt_tokens == 8


def test_gemini_stream_yields_chunks(monkeypatch):
    provider = GeminiProvider(model="gemini-1.5-flash")
    provider._api_key = "test-key"

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content_stream=lambda model, contents, config: [_FakeGeminiChunk("a"), _FakeGeminiChunk("b")]
        )
    )
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    chunks = list(provider.stream("hi"))
    assert chunks == ["a", "b"]


# --- Ollama ---


def test_ollama_generate_returns_result(monkeypatch):
    provider = OllamaProvider(model="llama3.1")

    fake_client = SimpleNamespace(
        chat=lambda model, messages, options: {
            "message": {"content": "Hi from Ollama"},
            "prompt_eval_count": 12,
            "eval_count": 6,
        }
    )
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    result = provider.generate("hi")
    assert result.text == "Hi from Ollama"
    assert result.provider == "ollama"
    assert result.prompt_tokens == 12


def test_ollama_stream_yields_chunks(monkeypatch):
    provider = OllamaProvider(model="llama3.1")

    fake_client = SimpleNamespace(
        chat=lambda model, messages, stream, options: [
            {"message": {"content": "a"}},
            {"message": {"content": "b"}},
            {"message": {}},  # no content -> should be skipped
        ]
    )
    monkeypatch.setattr(provider, "_client", lambda: fake_client)

    chunks = list(provider.stream("hi"))
    assert chunks == ["a", "b"]


def test_ollama_is_available_when_server_reachable(monkeypatch):
    provider = OllamaProvider(model="llama3.1")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: _FakeResponse())
    assert provider.is_available() is True
