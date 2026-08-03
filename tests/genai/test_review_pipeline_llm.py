"""Covers the LLM-dependent and optional-dependency branches of
genai/reviews/pipeline.py: KeyBERT enrichment (with fallback) and
LLM-based summarization (single batch, multi-batch reduce, and failure
fallback paths), using a fake LLM provider -- no network calls.
"""

from __future__ import annotations

from genai.llm.base import LLMProvider, LLMResult
from genai.reviews.pipeline import extract_keywords_keybert, extract_keywords_tfidf, summarize_reviews


class _FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, responses=None, fail_on=None):
        super().__init__(model="fake-model")
        self._responses = list(responses or [])
        self._fail_on = fail_on or set()
        self.calls = 0

    def is_available(self):
        return True

    def generate(self, prompt, system=None, **kwargs):
        self.calls += 1
        if self.calls in self._fail_on:
            raise RuntimeError("LLM call failed")
        text = self._responses[self.calls - 1] if self.calls - 1 < len(self._responses) else "summary"
        return LLMResult(text=text, provider=self.name, model=self.model)

    def stream(self, prompt, system=None, **kwargs):
        yield "n/a"


class _UnavailableLLM(_FakeLLM):
    def is_available(self):
        return False


def test_extract_keywords_keybert_falls_back_without_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "keybert":
            raise ImportError("no keybert installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    texts = ["clean room great staff", "clean room great location"]
    keywords = extract_keywords_keybert(texts, top_n=5)
    assert keywords == extract_keywords_tfidf(texts, top_n=5)


def test_extract_keywords_keybert_uses_model_when_available(monkeypatch):
    class _FakeKeyBERT:
        def extract_keywords(self, text, top_n=10, stop_words="english"):
            return [("clean", 0.9), ("staff", 0.8)]

    import sys
    import types

    fake_module = types.ModuleType("keybert")
    fake_module.KeyBERT = _FakeKeyBERT
    monkeypatch.setitem(sys.modules, "keybert", fake_module)

    keywords = extract_keywords_keybert(["clean room, great staff"], top_n=2)
    assert keywords == ["clean", "staff"]


def test_summarize_reviews_no_llm_falls_back():
    summary = summarize_reviews(["good"], ["positive"], llm_provider=None)
    assert "1 reviews" in summary


def test_summarize_reviews_llm_unavailable_falls_back():
    summary = summarize_reviews(["good"], ["positive"], llm_provider=_UnavailableLLM())
    assert "1 reviews" in summary


def test_summarize_reviews_single_batch_uses_llm_text():
    fake_llm = _FakeLLM(responses=["LLM summary of reviews."])
    summary = summarize_reviews(["good stay", "bad stay"], ["positive", "negative"], llm_provider=fake_llm)
    assert summary == "LLM summary of reviews."
    assert fake_llm.calls == 1


def test_summarize_reviews_multi_batch_reduces():
    texts = [f"review {i}" for i in range(250)]  # spans 2 batches of 200
    sentiments = ["positive"] * 250
    fake_llm = _FakeLLM(responses=["batch summary 1", "batch summary 2", "final combined summary"])
    summary = summarize_reviews(texts, sentiments, llm_provider=fake_llm)
    assert summary == "final combined summary"
    assert fake_llm.calls == 3  # 2 batch summaries + 1 reduce call


def test_summarize_reviews_batch_failure_falls_back_to_rule_based():
    fake_llm = _FakeLLM(fail_on={1})
    summary = summarize_reviews(["a", "b"], ["positive", "negative"], llm_provider=fake_llm)
    assert "2 reviews" in summary


def test_summarize_reviews_reduce_failure_concatenates_batches():
    texts = [f"review {i}" for i in range(250)]
    sentiments = ["positive"] * 250
    fake_llm = _FakeLLM(responses=["batch summary 1", "batch summary 2"], fail_on={3})
    summary = summarize_reviews(texts, sentiments, llm_provider=fake_llm)
    assert "batch summary 1" in summary and "batch summary 2" in summary
