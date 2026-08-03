"""End-to-end qa_chain tests using a fake/stub LLM provider and a FAISS
store seeded with a fake embedder -- fully offline."""

from __future__ import annotations

import numpy as np
import pytest

from genai.rag.chains.qa_chain import NoContextError, answer_question, stream_answer
from genai.rag.memory import SessionMemory
from genai.rag.retriever import HybridRetriever, MetadataFilter
from genai.rag.vector_store.faiss_store import FaissVectorStore
from genai.llm.base import LLMProvider, LLMResult


class FakeLLMProvider(LLMProvider):
    """Local test double conforming to the LLMProvider ABC -- avoids any
    network access. Kept local to this module to not depend on another
    test module's internals."""

    name = "fake"

    def __init__(self, model="fake-model", temperature=0.3, max_tokens=1024, available=True, text="fake answer"):
        super().__init__(model, temperature, max_tokens)
        self._available = available
        self._text = text

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt, system=None, **kwargs):
        return LLMResult(text=self._text, provider=self.name, model=self.model, prompt_tokens=10, completion_tokens=5, latency_ms=1.0)

    def stream(self, prompt, system=None, **kwargs):
        for word in self._text.split():
            yield word + " "


class FakeEmbedder:
    """Minimal fake embedder for qa_chain tests -- deterministic bag-of-words
    style vector so retrieval isn't purely random."""

    def embed_text(self, text: str):
        words = text.lower().split()
        vocab = ["occupancy", "revenue", "housekeeping", "pricing", "staff"]
        return [float(words.count(w)) + 0.01 for w in vocab]

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def seeded_store(tmp_path):
    embedder = FakeEmbedder()
    store = FaissVectorStore(store_dir=tmp_path)
    docs = [
        {"text": "Branch 1 occupancy forecast is 85% next week.", "source": "predict_occupancy", "doc_type": "prediction", "branch_id": 1},
        {"text": "Branch 1 revenue rose due to higher room rates.", "source": "mart_revenue_daily", "doc_type": "warehouse", "branch_id": 1},
    ]
    vectors = embedder.embed_batch([d["text"] for d in docs])
    store.add(vectors, docs)
    return store, embedder


def test_answer_question_empty_store_raises(tmp_path):
    store = FaissVectorStore(store_dir=tmp_path)
    retriever = HybridRetriever(store, embedder=FakeEmbedder())
    with pytest.raises(NoContextError):
        answer_question("What is occupancy?", retriever, None)


def test_answer_question_with_fake_llm_returns_citations(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)
    fake_llm = FakeLLMProvider(text="Occupancy is expected to be 85%.")
    result = answer_question("What is the occupancy forecast for branch 1?", retriever, fake_llm)
    assert result.used_llm is True
    assert result.answer == "Occupancy is expected to be 85%."
    assert len(result.citations) > 0
    assert result.citations[0].source in {"predict_occupancy", "mart_revenue_daily"}


def test_answer_question_without_llm_falls_back(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)
    result = answer_question("What is occupancy?", retriever, None)
    assert result.used_llm is False
    assert "No LLM provider" in result.answer or "don't have an LLM" in result.answer


def test_answer_question_llm_failure_falls_back(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)

    class BrokenLLM(FakeLLMProvider):
        def generate(self, prompt, system=None, **kwargs):
            raise RuntimeError("boom")

    result = answer_question("occupancy?", retriever, BrokenLLM())
    assert result.used_llm is False


def test_answer_question_with_memory_records_turns(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)
    memory = SessionMemory(max_turns=10, ttl_seconds=3600)
    fake_llm = FakeLLMProvider(text="answer")
    answer_question("occupancy?", retriever, fake_llm, session_id="s1", memory=memory)
    history = memory.get_history("s1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_answer_question_with_metadata_filter(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)
    fake_llm = FakeLLMProvider(text="filtered answer")
    flt = MetadataFilter(doc_type="prediction")
    result = answer_question("occupancy?", retriever, fake_llm, metadata_filter=flt)
    assert all(c.doc_type == "prediction" for c in result.citations)


def test_stream_answer_empty_store_raises(tmp_path):
    store = FaissVectorStore(store_dir=tmp_path)
    retriever = HybridRetriever(store, embedder=FakeEmbedder())
    with pytest.raises(NoContextError):
        list(stream_answer("q", retriever, None))


def test_stream_answer_with_fake_llm_yields_chunks(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)
    fake_llm = FakeLLMProvider(text="streamed words here")
    chunks = list(stream_answer("occupancy?", retriever, fake_llm))
    assert "".join(chunks).strip() == "streamed words here"


def test_stream_answer_without_llm_yields_fallback(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)
    chunks = list(stream_answer("occupancy?", retriever, None))
    assert len(chunks) == 1


def test_stream_answer_llm_failure_falls_back(seeded_store):
    store, embedder = seeded_store
    retriever = HybridRetriever(store, embedder=embedder)

    class BrokenLLM(FakeLLMProvider):
        def stream(self, prompt, system=None, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    chunks = list(stream_answer("occupancy?", retriever, BrokenLLM()))
    assert len(chunks) == 1
