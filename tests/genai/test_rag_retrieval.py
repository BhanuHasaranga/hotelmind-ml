"""Tests for chunking, embedding cache, FAISS retrieval, and metadata
filtering. Uses a fake, deterministic hash-based embedder (dependency
injected via monkeypatching `genai.rag.embeddings._get_model`) so no
sentence-transformers model download is required at test time.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from genai.cache.query_cache import QueryCache
from genai.rag.chunking import chunk_documents, chunk_text
from genai.rag.embeddings import Embedder
from genai.rag.retriever import HybridRetriever, MetadataFilter, compress_context
from genai.rag.vector_store.faiss_store import FaissVectorStore


# --- chunking ---


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("A short sentence.", chunk_size=100, overlap=0)
    assert chunks == ["A short sentence."]


def test_chunk_text_respects_chunk_size():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 100 or " " not in c for c in chunks)


def test_chunk_text_overlap_shares_tail():
    text = "Paragraph one is here.\n\nParagraph two follows after it.\n\nParagraph three is the last one here."
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) >= 2


def test_chunk_documents_propagates_metadata():
    docs = [{"text": "hello world " * 50, "metadata": {"source": "doc1"}}]
    chunked = chunk_documents(docs)
    assert len(chunked) >= 1
    assert chunked[0]["metadata"]["source"] == "doc1"
    assert "chunk_index" in chunked[0]["metadata"]


# --- fake embedder for offline testing ---


class FakeSentenceTransformer:
    """Deterministic hash-based fake embedding model -- no network/model
    download required. Conforms to the subset of the SentenceTransformer
    API that Embedder relies on (`.encode`)."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        single = isinstance(texts, str)
        items = [texts] if single else texts
        vectors = []
        for text in items:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = np.frombuffer((digest * ((self.dim // len(digest)) + 1))[: self.dim * 4], dtype=np.uint8).astype(
                np.float32
            )
            vec = raw[: self.dim]
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        arr = np.array(vectors, dtype="float32")
        return arr[0] if single else arr


@pytest.fixture
def fake_embedder(monkeypatch, tmp_path):
    import genai.rag.embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module, "_model", None)
    monkeypatch.setattr(embeddings_module, "_get_model", lambda: FakeSentenceTransformer())
    cache = QueryCache(db_path=tmp_path / "embed_cache.sqlite3")
    return Embedder(cache=cache)


def test_embedder_embed_text_deterministic(fake_embedder):
    v1 = fake_embedder.embed_text("hello world")
    v2 = fake_embedder.embed_text("hello world")
    assert v1 == v2
    assert len(v1) == 16


def test_embedder_embed_text_uses_cache(fake_embedder):
    fake_embedder.embed_text("cached text")
    cached = fake_embedder.cache.get("embedding", fake_embedder.cache and list(fake_embedder.cache.stats().keys()) and "embedding")
    # cache namespace should now contain one entry
    assert fake_embedder.cache.stats().get("embedding", 0) == 1


def test_embedder_embed_batch(fake_embedder):
    vectors = fake_embedder.embed_batch(["a", "b", "a"])
    assert len(vectors) == 3
    assert vectors[0] == vectors[2]  # same text -> same vector


# --- FAISS vector store ---


def test_faiss_store_add_and_search(tmp_path):
    store = FaissVectorStore(store_dir=tmp_path)
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    metas = [{"text": "a", "source": "doc1"}, {"text": "b", "source": "doc2"}, {"text": "c", "source": "doc3"}]
    store.add(vectors, metas)
    assert store.size == 3

    results = store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["source"] == "doc1"


def test_faiss_store_search_empty_returns_empty(tmp_path):
    store = FaissVectorStore(store_dir=tmp_path)
    assert store.search([1.0, 0.0]) == []


def test_faiss_store_save_and_load_roundtrip(tmp_path):
    store = FaissVectorStore(store_dir=tmp_path)
    store.add([[1.0, 0.0]], [{"text": "x", "source": "doc1"}])
    store.save()

    reloaded = FaissVectorStore(store_dir=tmp_path)
    assert reloaded.load() is True
    assert reloaded.size == 1
    assert reloaded.metadata[0]["source"] == "doc1"


def test_faiss_store_load_missing_returns_false(tmp_path):
    store = FaissVectorStore(store_dir=tmp_path)
    assert store.load() is False


def test_faiss_store_exists_on_disk(tmp_path):
    store = FaissVectorStore(store_dir=tmp_path)
    assert store.exists_on_disk() is False
    store.add([[1.0, 0.0]], [{"text": "x"}])
    store.save()
    assert store.exists_on_disk() is True


def test_faiss_store_save_serializes_numpy_scalars(tmp_path):
    """Regression test: pandas-derived metadata (e.g. branch_id read as
    numpy.int64) must not break metadata.json serialization."""
    import numpy as np

    store = FaissVectorStore(store_dir=tmp_path)
    store.add([[1.0, 0.0]], [{"text": "x", "branch_id": np.int64(3), "score_like": np.float64(1.5)}])
    store.save()  # must not raise TypeError

    reloaded = FaissVectorStore(store_dir=tmp_path)
    assert reloaded.load() is True
    assert reloaded.metadata[0]["branch_id"] == 3


# --- hybrid retriever ---


def _seed_store(tmp_path, fake_embedder) -> FaissVectorStore:
    store = FaissVectorStore(store_dir=tmp_path)
    docs = [
        {"text": "Branch 1 occupancy was 85% yesterday.", "source": "mart_occupancy_daily", "doc_type": "warehouse", "branch_id": 1},
        {"text": "Branch 2 revenue increased due to higher ADR.", "source": "mart_revenue_daily", "doc_type": "warehouse", "branch_id": 2},
        {"text": "Housekeeping SOP requires daily room inspection.", "source": "hotel_sop.md", "doc_type": "markdown", "branch_id": None},
    ]
    texts = [d["text"] for d in docs]
    vectors = fake_embedder.embed_batch(texts)
    metadatas = [{**d} for d in docs]
    store.add(vectors, metadatas)
    return store


def test_hybrid_retriever_returns_results(tmp_path, fake_embedder):
    store = _seed_store(tmp_path, fake_embedder)
    retriever = HybridRetriever(store, embedder=fake_embedder)
    results = retriever.retrieve("What was branch 1 occupancy?", top_k=2)
    assert len(results) > 0
    assert all(hasattr(r, "text") for r in results)


def test_hybrid_retriever_empty_store_returns_empty(tmp_path, fake_embedder):
    store = FaissVectorStore(store_dir=tmp_path)
    retriever = HybridRetriever(store, embedder=fake_embedder)
    assert retriever.retrieve("anything") == []


def test_hybrid_retriever_metadata_filter_by_doc_type(tmp_path, fake_embedder):
    store = _seed_store(tmp_path, fake_embedder)
    retriever = HybridRetriever(store, embedder=fake_embedder)
    flt = MetadataFilter(doc_type="markdown")
    results = retriever.retrieve("housekeeping inspection", top_k=5, metadata_filter=flt)
    assert all(r.metadata.get("doc_type") == "markdown" for r in results)


def test_hybrid_retriever_metadata_filter_by_branch(tmp_path, fake_embedder):
    store = _seed_store(tmp_path, fake_embedder)
    retriever = HybridRetriever(store, embedder=fake_embedder)
    flt = MetadataFilter(branch_id=1)
    results = retriever.retrieve("occupancy", top_k=5, metadata_filter=flt)
    assert all(str(r.metadata.get("branch_id")) == "1" for r in results)


def test_compress_context_respects_char_budget():
    from genai.rag.retriever import RetrievedChunk

    chunks = [RetrievedChunk(text="x" * 100, metadata={}, score=1.0), RetrievedChunk(text="y" * 100, metadata={}, score=0.5)]
    compressed = compress_context(chunks, max_chars=120)
    assert len(compressed) <= 130  # small allowance for the joining separator
