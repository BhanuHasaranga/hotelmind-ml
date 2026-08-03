"""Builds/rebuilds the FAISS index from documents + warehouse marts +
prediction snapshots. Supports full rebuild and incremental indexing
(content-hash tracked, new/changed documents only).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from genai.config.genai_settings import genai_settings
from genai.rag.chunking import chunk_documents
from genai.rag.embeddings import get_embedder
from genai.rag.loaders.csv_loader import load_csv
from genai.rag.loaders.markdown_loader import load_markdown
from genai.rag.loaders.pdf_loader import load_pdf
from genai.rag.loaders.prediction_loader import load_all_prediction_documents
from genai.rag.loaders.txt_loader import load_txt
from genai.rag.loaders.warehouse_loader import load_all_warehouse_documents
from genai.rag.vector_store.faiss_store import FaissVectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MANIFEST_FILENAME = "indexed_manifest.json"

_LOADERS = {
    ".md": load_markdown,
    ".txt": load_txt,
    ".csv": load_csv,
    ".pdf": load_pdf,
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_document_dir() -> tuple[list[dict], dict[str, str]]:
    """Loads all supported documents under `document_dir_path`. Returns
    (documents, {relative_path: content_hash}) for incremental tracking."""
    documents = []
    hashes = {}
    doc_dir = genai_settings.document_dir_path
    for path in sorted(doc_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _LOADERS:
            continue
        loader = _LOADERS[path.suffix.lower()]
        try:
            documents.extend(loader(path))
            hashes[str(path.relative_to(doc_dir))] = _file_hash(path)
        except Exception as exc:
            logger.warning("Failed to load document %s: %s", path, exc)
    return documents, hashes


def _load_manifest() -> dict[str, str]:
    path = genai_settings.vector_db_dir_path / _MANIFEST_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(hashes: dict[str, str]) -> None:
    path = genai_settings.vector_db_dir_path / _MANIFEST_FILENAME
    path.write_text(json.dumps(hashes), encoding="utf-8")


def build_index(include_warehouse: bool = True, include_predictions: bool = True) -> FaissVectorStore:
    """Full rebuild: re-embeds and re-indexes every document, warehouse mart
    snapshot, and prediction snapshot."""
    documents, hashes = _load_document_dir()

    if include_warehouse:
        documents.extend(load_all_warehouse_documents())
    if include_predictions:
        try:
            documents.extend(load_all_prediction_documents())
        except Exception as exc:
            logger.warning("Skipping prediction documents in index build: %s", exc)

    chunks = chunk_documents(documents)
    logger.info("build_index: %d source documents -> %d chunks", len(documents), len(chunks))

    store = FaissVectorStore()
    if chunks:
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed_batch(texts)
        metadatas = [{**c["metadata"], "text": c["text"]} for c in chunks]
        store.add(vectors, metadatas)

    store.save()
    _save_manifest(hashes)
    return store


def incremental_index(include_warehouse: bool = True, include_predictions: bool = True) -> FaissVectorStore:
    """Incremental rebuild: only recomputes chunks/embeddings for
    new-or-changed documents (tracked via content hash), then re-adds
    warehouse/prediction snapshots fresh (those are always regenerated,
    since they represent point-in-time data rather than static files)."""
    previous_hashes = _load_manifest()
    documents, current_hashes = _load_document_dir()

    changed_paths = {
        path for path, h in current_hashes.items() if previous_hashes.get(path) != h
    }
    removed_paths = set(previous_hashes) - set(current_hashes)

    store = FaissVectorStore()
    loaded_existing = store.load()

    if not loaded_existing or changed_paths or removed_paths:
        logger.info(
            "incremental_index: %d changed, %d removed documents; performing full rebuild "
            "(FAISS flat index does not support in-place row deletion)",
            len(changed_paths),
            len(removed_paths),
        )
        return build_index(include_warehouse=include_warehouse, include_predictions=include_predictions)

    logger.info("incremental_index: no document changes detected; refreshing warehouse/prediction snapshots only")
    fresh_docs = []
    if include_warehouse:
        fresh_docs.extend(load_all_warehouse_documents())
    if include_predictions:
        try:
            fresh_docs.extend(load_all_prediction_documents())
        except Exception as exc:
            logger.warning("Skipping prediction documents in incremental index: %s", exc)

    if fresh_docs:
        chunks = chunk_documents(fresh_docs)
        embedder = get_embedder()
        vectors = embedder.embed_batch([c["text"] for c in chunks])
        metadatas = [{**c["metadata"], "text": c["text"]} for c in chunks]
        store.add(vectors, metadatas)
        store.save()

    return store


def get_or_build_store() -> FaissVectorStore:
    store = FaissVectorStore()
    if store.load():
        return store
    return build_index()


def index_stats() -> dict:
    store = FaissVectorStore()
    loaded = store.load()
    return {
        "indexed": loaded and store.size > 0,
        "vector_count": store.size,
        "store_dir": str(genai_settings.vector_db_dir_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild:
        build_index()
    else:
        incremental_index()
