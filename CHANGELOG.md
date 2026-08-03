# Changelog

All notable changes to this project are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to phase completions in the HotelMind AI roadmap (see
README.md "Roadmap").

## [Unreleased]

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
