"""Tests for genai/rag/loaders/* and genai/rag/indexer.py, fully offline
(fake embedder, local temp documents, stubbed warehouse/prediction loaders).
"""

from __future__ import annotations

import numpy as np
import pytest

from genai.config.genai_settings import genai_settings
from genai.rag import indexer as indexer_module
from genai.rag.loaders.csv_loader import load_csv
from genai.rag.loaders.markdown_loader import load_markdown
from genai.rag.loaders.pdf_loader import load_pdf
from genai.rag.loaders.prediction_loader import load_all_prediction_documents, load_occupancy_predictions
from genai.rag.loaders.txt_loader import load_txt
from genai.rag.loaders.warehouse_loader import load_all_warehouse_documents, load_warehouse_documents


# --- txt/markdown/csv loaders ---


def test_load_txt(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")
    docs = load_txt(path)
    assert docs[0]["text"] == "hello world"
    assert docs[0]["metadata"]["doc_type"] == "txt"


def test_load_markdown_with_headings(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nIntro text.\n\n## Section\n\nBody text.", encoding="utf-8")
    docs = load_markdown(path)
    assert len(docs) == 2
    assert docs[0]["metadata"]["heading"] == "Title"
    assert docs[1]["metadata"]["heading"] == "Section"


def test_load_markdown_no_headings(tmp_path):
    path = tmp_path / "plain.md"
    path.write_text("just plain text, no headings", encoding="utf-8")
    docs = load_markdown(path)
    assert len(docs) == 1
    assert "heading" not in docs[0]["metadata"]


def test_load_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    docs = load_csv(path)
    assert len(docs) == 2
    assert "a=1" in docs[0]["text"]


def test_load_csv_max_rows(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    docs = load_csv(path, max_rows=2)
    assert len(docs) == 2


def test_load_pdf(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    docs = load_pdf(path)
    # Blank page has no extractable text, so this should not error, just possibly empty.
    assert isinstance(docs, list)


# --- warehouse loader ---


def test_load_warehouse_documents_unknown_mart_raises():
    with pytest.raises(ValueError):
        load_warehouse_documents("mart_not_real")


def test_load_warehouse_documents_uses_local_source(monkeypatch):
    docs = load_warehouse_documents("mart_occupancy_daily", max_rows=5)
    assert len(docs) > 0
    assert docs[0]["metadata"]["doc_type"] == "warehouse"
    assert "occupancy" in docs[0]["text"]


def test_load_all_warehouse_documents_skips_failures(monkeypatch):
    import genai.rag.loaders.warehouse_loader as wl

    def _boom(mart_name, max_rows=200):
        if mart_name == "mart_staff_daily":
            raise RuntimeError("boom")
        return [{"text": "x", "metadata": {"source": mart_name, "doc_type": "warehouse"}}]

    monkeypatch.setattr(wl, "load_warehouse_documents", _boom)
    docs = wl.load_all_warehouse_documents()
    assert len(docs) == 3  # 4 marts minus the one that raised


# --- prediction loader ---


def test_load_occupancy_predictions_handles_failure(monkeypatch):
    import genai.rag.loaders.prediction_loader as pl

    def _boom(branch_id):
        raise RuntimeError("model not trained")

    monkeypatch.setattr(pl, "forecast_occupancy", _boom)
    docs = pl.load_occupancy_predictions(branch_ids=[1, 2])
    assert docs == []


def test_load_occupancy_predictions_success(monkeypatch):
    import pandas as pd

    import genai.rag.loaders.prediction_loader as pl

    def _fake_forecast(branch_id):
        return pd.DataFrame(
            [
                {
                    "occupancy_date": "2026-08-05",
                    "predicted_occupancy_pct": 70.0,
                    "ci_lower": 65.0,
                    "ci_upper": 75.0,
                    "model_used": "prophet",
                }
            ]
        )

    monkeypatch.setattr(pl, "forecast_occupancy", _fake_forecast)
    docs = pl.load_occupancy_predictions(branch_ids=[1])
    assert len(docs) == 1
    assert "70.0%" in docs[0]["text"]


def test_load_all_prediction_documents_delegates(monkeypatch):
    import genai.rag.loaders.prediction_loader as pl

    monkeypatch.setattr(pl, "load_occupancy_predictions", lambda branch_ids=None: [{"text": "x", "metadata": {}}])
    docs = load_all_prediction_documents()
    assert docs == [{"text": "x", "metadata": {}}]


# --- indexer ---


class _FakeSentenceTransformer:
    def encode(self, texts, normalize_embeddings=True):
        single = isinstance(texts, str)
        items = [texts] if single else texts
        vectors = [np.random.default_rng(abs(hash(t)) % (2**32)).random(8).astype("float32") for t in items]
        arr = np.array(vectors)
        return arr[0] if single else arr


@pytest.fixture
def indexer_env(tmp_path, monkeypatch):
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()
    (doc_dir / "policy.md").write_text("# Policy\n\nCheck-in is at 2pm.", encoding="utf-8")
    (doc_dir / "notes.txt").write_text("Some plain text notes about hotel operations.", encoding="utf-8")

    vector_dir = tmp_path / "vector_store"

    monkeypatch.setattr(genai_settings, "DOCUMENT_DIR", str(doc_dir))
    monkeypatch.setattr(type(genai_settings), "document_dir_path", property(lambda self: doc_dir), raising=False)
    monkeypatch.setattr(type(genai_settings), "vector_db_dir_path", property(lambda self: vector_dir), raising=False)

    import genai.rag.embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module, "_model", None)
    monkeypatch.setattr(embeddings_module, "_get_model", lambda: _FakeSentenceTransformer())

    monkeypatch.setattr(indexer_module, "load_all_warehouse_documents", lambda: [])
    monkeypatch.setattr(indexer_module, "load_all_prediction_documents", lambda: [])

    return doc_dir, vector_dir


def test_build_index_creates_store(indexer_env):
    doc_dir, vector_dir = indexer_env
    store = indexer_module.build_index(include_warehouse=False, include_predictions=False)
    assert store.size > 0
    assert (vector_dir / "index.faiss").exists()
    assert (vector_dir / "indexed_manifest.json").exists()


def test_index_stats_reflects_build(indexer_env):
    indexer_module.build_index(include_warehouse=False, include_predictions=False)
    stats = indexer_module.index_stats()
    assert stats["indexed"] is True
    assert stats["vector_count"] > 0


def test_index_stats_empty_before_build(indexer_env):
    stats = indexer_module.index_stats()
    assert stats["indexed"] is False


def test_get_or_build_store_builds_if_missing(indexer_env):
    store = indexer_module.get_or_build_store()
    assert store.size > 0


def test_get_or_build_store_loads_existing(indexer_env):
    indexer_module.build_index(include_warehouse=False, include_predictions=False)
    store = indexer_module.get_or_build_store()
    assert store.size > 0


def test_incremental_index_full_rebuild_when_no_manifest(indexer_env):
    store = indexer_module.incremental_index(include_warehouse=False, include_predictions=False)
    assert store.size > 0


def test_incremental_index_refresh_only_when_unchanged(indexer_env):
    doc_dir, vector_dir = indexer_env
    indexer_module.build_index(include_warehouse=False, include_predictions=False)
    # Nothing changed -> should take the "refresh warehouse/prediction only" path
    store = indexer_module.incremental_index(include_warehouse=False, include_predictions=False)
    assert store.size > 0


def test_incremental_index_rebuilds_on_change(indexer_env):
    doc_dir, vector_dir = indexer_env
    indexer_module.build_index(include_warehouse=False, include_predictions=False)
    (doc_dir / "policy.md").write_text("# Policy\n\nCheck-in is at 3pm now.", encoding="utf-8")
    store = indexer_module.incremental_index(include_warehouse=False, include_predictions=False)
    assert store.size > 0
