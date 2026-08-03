"""FAISS index persistence: build, load, add, search, and save, with a
`metadata.json` sidecar mapping vector row -> chunk text + metadata (used
for retrieval citations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from genai.config.genai_settings import genai_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


def _json_default(obj):
    """Handles numpy scalar types (e.g. branch_id read from a pandas
    DataFrame as numpy.int64) that plain `json.dumps` can't serialize."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class FaissVectorStore:
    def __init__(self, dim: int | None = None, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or genai_settings.vector_db_dir_path
        self.dim = dim
        self._index = None
        self.metadata: list[dict] = []

    @property
    def index_path(self) -> Path:
        return self.store_dir / INDEX_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.store_dir / METADATA_FILENAME

    def _ensure_index(self, dim: int) -> None:
        import faiss

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
            self.dim = dim

    def add(self, vectors: list[list[float]], metadatas: list[dict]) -> None:
        if not vectors:
            return
        arr = np.asarray(vectors, dtype="float32")
        self._ensure_index(arr.shape[1])
        self._index.add(arr)
        self.metadata.extend(metadatas)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        arr = np.asarray([query_vector], dtype="float32")
        scores, indices = self._index.search(arr, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    def save(self) -> None:
        import faiss

        self.store_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, default=_json_default), encoding="utf-8")
        logger.info("Saved FAISS index (%d vectors) to %s", self.size, self.store_dir)

    def load(self) -> bool:
        import faiss

        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        self._index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.dim = self._index.d
        return True

    @property
    def size(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    def exists_on_disk(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()
