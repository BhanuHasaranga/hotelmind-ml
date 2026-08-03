# Changelog

All notable changes to this project are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to phase completions in the HotelMind AI roadmap (see
README.md "Roadmap").

## [Unreleased]

### Added — Phase 5: Generative AI Layer
- `genai/` subpackage: LLM provider abstraction (`llm/base.py`, `openai_provider.py`, `gemini_provider.py`, `ollama_provider.py`, `factory.py`) selected purely by `LLM_PROVIDER` config, with token-usage/latency logging (`observability/llm_logging.py`) and a SQLite-backed embedding/query cache (`cache/query_cache.py`).
- `genai/data_access/` — `DATA_SOURCE=postgres|local` mart abstraction; local path synthesizes small mart-equivalent snapshots under `data/warehouse/` if the live warehouse marts aren't populated.
- Module 1 — Guest Review Analysis: `genai/reviews/synthetic_reviews.py` (seeded, English-majority with Sinhala/Tamil samples, 50k-100k rows), `pipeline.py` (sentiment, emotion, complaint detection, LDA topics, TF-IDF/KeyBERT keywords, LLM summarization, CSAT scoring, trend detection), `service.py` (precompute-then-query pattern).
- Module 2 — Hotel AI Assistant (RAG): loaders (PDF/Markdown/CSV/TXT/warehouse/prediction), `chunking.py`, `embeddings.py`, FAISS `vector_store/` with citation metadata, hybrid dense+BM25 `retriever.py`, `memory.py` (session memory), `chains/qa_chain.py` (retrieval → LLM → cited answer), `indexer.py` (full + incremental rebuild), SSE streaming support. Real starter documents: `genai/rag/documents/hotel_policies.md`, `hotel_sop.md`.
- Module 3 — AI Insights Generator: rule modules per category (revenue, occupancy, pricing, guest_experience, restaurant_waste, staff, churn, anomaly), `scoring.py`/`priority.py`, `service.py` producing structured JSON findings with LLM-generated (or rule-based fallback) recommendations.
- Versioned prompts (`genai/prompts/system/*.md`) + `loader.py`.
- New API routers: `api/routers/reviews.py`, `api/routers/rag.py` (incl. SSE `/rag/query`), `api/routers/insights.py`; `api/schemas_genai.py`; wired into `api/main.py` with a graceful, non-failing RAG index warm-up at startup.
- `tests/genai/` — 300+ new tests (offline, fake LLM providers, fake embedder), ≥90% coverage on `genai/`.
- `docs/phase5.md` — architecture, RAG/insights sequence diagrams, LLM provider class diagram, full API reference, deployment notes.
- New dependencies: `langchain`, `langchain-community`, `faiss-cpu`, `sentence-transformers`, `openai`, `google-genai`, `ollama`, `pypdf`, `keybert`, `sse-starlette`, `rank-bm25`.

### Added — Phase 4 Release Preparation
- `demo/` — sample API requests/responses for all 5 prediction endpoints, generated from the live API against trained models.
- `docs/api/` — per-endpoint API reference (purpose, request/response schema, validation, errors, curl example).
- `docs/architecture/` — Mermaid diagrams: system overview, Phase 4 pipeline, prediction flow, training pipeline.
- `docs/models/` — model cards for all 5 domains (purpose, features, algorithm, metrics, limitations).
- `docs/datasets/` — canonical dataset, warehouse transformation, and synthetic data documentation.
- `docs/demo/screenshots_required.md` — portfolio screenshot checklist.
- `scripts/train_all.py`, `scripts/predict_examples.py`, `scripts/verify_phase4.py` — reproducibility scripts.
- `reports/final_release/phase4_release_report.md` — release readiness assessment.
- `LICENSE` (MIT), this `CHANGELOG.md`.

## [0.4.0] — Phase 4: Machine Learning

### Added
- Feature engineering pipelines for Occupancy, Pricing, Restaurant, Staffing, and Churn (`src/pipelines/feature_engineering.py`), producing `data/features/*.parquet` from warehouse parquet only — no raw CSVs, no live Postgres dependency.
- `src/features/occupancy_aggregation.py` — derives a daily occupancy/revenue "mart" directly from `fact_booking`, since no pre-aggregated mart exists in the warehouse.
- `src/pipelines/synthetic_data.py` — deterministic, occupancy-driven synthetic seed generation for Restaurant and Staffing (no real data exists for either domain anywhere in the project).
- Trained models for all 5 domains: Occupancy (XGBoost, Prophet), Pricing (XGBoost), Restaurant (3× XGBoost, per meal), Staffing (GradientBoostingRegressor), Churn (RandomForest, XGBoost).
- `src/models/occupancy/lstm_model.py` and `src/models/staffing/or_tools_scheduler.py` — intentional `NotImplementedError` scaffolds, documented as future work.
- FastAPI prediction service (`api/main.py`) with `POST /predict/{occupancy,pricing,restaurant,staff,churn}`, verified live against trained models.
- `reports/features/`, `reports/models/`, `reports/model_discovery/`, `reports/final_phase4/` — feature/model documentation and evaluation reports.
- 75 pytest tests, fully offline (no live database dependency).

### Changed
- Reworked all 5 pipelines' data loading to read exclusively from `data/processed/` and `data/warehouse/` parquet — the pre-existing scaffold assumed a live Postgres warehouse with pre-aggregated marts that were never populated.
- `src/prediction/predict_churn.py` and `predict_occupancy.py` — fixed date-anchoring to use the dataset's own date range (2015–2017) instead of the real system clock, which previously produced a degenerate churn label and nonsensical occupancy forecasts.

### Fixed
- `src/pipelines/churn_pipeline.py::clean()` — removed a reference to a column not yet created at that pipeline stage.

## [0.3.0] — Phase 3: Data Engineering

### Added
- `src/pipelines/data_cleaning.py` — full cleaning pipeline for the canonical `hotel_bookings.csv` (dedup, null normalization, country/ADR/adults validation, datatype conversion, datetime parsing, categorical normalization). Output: `data/processed/hotel_bookings_clean.parquet`.
- `src/pipelines/keygen.py` — deterministic surrogate key generation (hashing/fixed-lookup, no randomness) for `reservation_id`, `guest_id`, `hotel_id`, `branch_id`, `room_type_id`, `date_id`.
- `src/pipelines/warehouse_loader.py` — transforms the cleaned dataset into a star-schema warehouse (`dim_date`, `dim_hotel`, `dim_guest`, `dim_room_type`, `fact_booking`), written locally to `data/warehouse/`, with an optional `--write-db` flag for a live Postgres load.
- `reports/data_discovery/` — dataset inventory, quality report, schema diagram, warehouse mapping, cleaning plan.
- `reports/warehouse_loading/` — loading summary, key generation strategy, validation report, mapping/assumption summary, data lineage report.
- Warehouse loader tests (deterministic keys, validation logic, mocked database writes — no live DB dependency).

## [0.2.0] — Phase 2: Hotel Management System

Hotel booking/management system scope (outside this repository's direct
history — the earliest commits in this repo begin with the ML/data-engineering
scaffold; Phase 2 details are tracked in the companion project).

## [0.1.0] — Phase 1: Infrastructure

Initial project scaffold: `src/config`, `src/database`, `src/utils/logging`,
`api/` skeleton, `.env.example`, `requirements.txt`, project folder structure.
