"""Hybrid retriever: blends dense FAISS similarity with sparse BM25 keyword
scoring, plus metadata filtering (hotel_id / doc_type / date range) and a
simple context-compression step (trims retrieved chunks to a token budget).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from genai.config.genai_settings import genai_settings
from genai.rag.embeddings import Embedder, get_embedder
from genai.rag.vector_store.faiss_store import FaissVectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float


@dataclass
class MetadataFilter:
    doc_type: str | None = None
    branch_id: object | None = None
    date_from: str | None = None
    date_to: str | None = None


def _passes_filter(metadata: dict, flt: MetadataFilter | None) -> bool:
    if flt is None:
        return True
    if flt.doc_type and metadata.get("doc_type") != flt.doc_type:
        return False
    if flt.branch_id is not None and str(metadata.get("branch_id")) != str(flt.branch_id):
        return False
    date = metadata.get("date")
    if flt.date_from and date and date < flt.date_from:
        return False
    if flt.date_to and date and date > flt.date_to:
        return False
    return True


class HybridRetriever:
    def __init__(self, store: FaissVectorStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        self._build_bm25()

    def _build_bm25(self) -> None:
        if not self.store.metadata:
            self._bm25 = None
            return
        self._corpus_tokens = [m.get("text", "").lower().split() for m in self.store.metadata]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def refresh(self) -> None:
        self._build_bm25()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        dense_weight: float = 0.6,
    ) -> list[RetrievedChunk]:
        top_k = top_k or genai_settings.RETRIEVER_TOP_K
        if self.store.size == 0:
            return []

        query_vector = self.embedder.embed_text(query)
        dense_results = self.store.search(query_vector, top_k=max(top_k * 3, top_k))

        sparse_scores: dict[int, float] = {}
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
            for idx, meta in enumerate(self.store.metadata):
                sparse_scores[idx] = scores[idx] / max_score

        combined: list[RetrievedChunk] = []
        seen_texts = set()
        for entry in dense_results:
            meta = {k: v for k, v in entry.items() if k not in ("score", "text")}
            text = entry.get("text", "")
            if not _passes_filter(meta, metadata_filter):
                continue
            # find matching index in store.metadata for sparse score lookup
            try:
                idx = self.store.metadata.index({**meta, "text": text})
            except ValueError:
                idx = None
            sparse = sparse_scores.get(idx, 0.0) if idx is not None else 0.0
            blended = dense_weight * entry["score"] + (1 - dense_weight) * sparse
            if text in seen_texts:
                continue
            seen_texts.add(text)
            combined.append(RetrievedChunk(text=text, metadata=meta, score=blended))

        combined.sort(key=lambda c: c.score, reverse=True)
        return combined[:top_k]


def compress_context(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    """Trims retrieved chunks down to a character budget before sending to
    the LLM, preferring the highest-scoring chunks first."""
    parts = []
    total = 0
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        remaining = max_chars - total
        if remaining <= 0:
            break
        text = chunk.text[:remaining]
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)
