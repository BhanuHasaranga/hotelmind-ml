"""SentenceTransformers embedding wrapper with an on-disk content-hash-keyed
cache (`genai.cache.query_cache`) to avoid recomputing embeddings for
unchanged text.
"""

from __future__ import annotations

import hashlib

import numpy as np

from genai.cache.query_cache import QueryCache, get_cache
from genai.config.genai_settings import genai_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_NAMESPACE = "embedding"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(genai_settings.EMBEDDING_MODEL)
    return _model


def _content_hash(text: str) -> str:
    return hashlib.sha256(f"{genai_settings.EMBEDDING_MODEL}:{text}".encode("utf-8")).hexdigest()


class Embedder:
    """Wraps a SentenceTransformers model with a disk cache."""

    def __init__(self, cache: QueryCache | None = None) -> None:
        self.cache = cache or get_cache()

    def embed_text(self, text: str) -> list[float]:
        key = _content_hash(text)
        cached = self.cache.get(_NAMESPACE, key)
        if cached is not None:
            return cached
        vector = _get_model().encode(text, normalize_embeddings=True).tolist()
        self.cache.set(_NAMESPACE, key, vector)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        to_compute_idx = []
        to_compute_text = []

        for i, text in enumerate(texts):
            key = _content_hash(text)
            cached = self.cache.get(_NAMESPACE, key)
            if cached is not None:
                results[i] = cached
            else:
                to_compute_idx.append(i)
                to_compute_text.append(text)

        if to_compute_text:
            vectors = _get_model().encode(to_compute_text, normalize_embeddings=True)
            for idx, text, vector in zip(to_compute_idx, to_compute_text, vectors):
                vector_list = np.asarray(vector).tolist()
                results[idx] = vector_list
                self.cache.set(_NAMESPACE, _content_hash(text), vector_list)

        return results  # type: ignore[return-value]


_default_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder
