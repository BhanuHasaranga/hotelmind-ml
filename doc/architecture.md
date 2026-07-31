# Architecture

Phase 4 of HotelMind AI builds five ML modules — occupancy forecasting,
dynamic pricing, restaurant demand, staff optimization, and customer churn —
reading **only** from local parquet files: `data/processed/hotel_bookings_
clean.parquet` (Phase 3 cleaned dataset) and `data/warehouse/{dim_date,
dim_hotel,dim_guest,dim_room_type,fact_booking}.parquet` (Phase 3 warehouse
loader output). No live Postgres connection is required anywhere in this
phase — the earlier design (querying `sql/*.sql` against a `hotelmind_
warehouse` Postgres schema populated by a separate `hotelmind-data` project)
was scaffolded but never had real data behind it, so it has been replaced.

Out of scope for this phase: LLMs, LangChain, RAG, MLflow, Kafka, AWS,
Terraform, Docker changes, monitoring. Those belong to later phases.

## Pipeline shape

Every module follows the same sequence:

```
Load (parquet via src/features, pandas joins/aggregation) -> Clean -> Feature Engineer -> Train -> Evaluate -> Save -> Predict
```

This is implemented once as `src/pipelines/base_pipeline.py::BasePipeline`
and subclassed per domain. Every trained model — regardless of algorithm —
implements the same `BaseMLModel` interface (`src/models/base.py`): `train`,
`evaluate`, `predict`, `save`, `load`. This is what lets five different
domains (time series, regression, classification, Prophet, XGBoost,
scikit-learn) share one pipeline shape without duplicating boilerplate.

```
        data/warehouse/*.parquet          data/processed/*.parquet
                  │                                 │
                  ▼                                 ▼
   src/features/occupancy_aggregation.py   (churn reads dim_guest +
   (derives daily occupancy_pct/revenue     fact_booking directly)
    from raw per-booking fact_booking)
                  │
     ┌────────────┼─────────────────────────┐
     ▼            ▼                         ▼
 Occupancy    Pricing            src/pipelines/synthetic_data.py
 Pipeline     Pipeline           (Restaurant/Staffing: synthetic daily
                                   seeds, driven by real occupancy)
                  │
                  ▼
        ┌───────────────────┐
        │   BasePipeline     │  load -> clean -> engineer_features
        │  (per-domain impl) │  -> split -> train -> evaluate -> save
        └─────────┬──────────┘
                   │
      ┌────────────┼─────────────┐
      ▼            ▼             ▼
BaseMLModel   reports/*.json   models/*.pkl
(train/evaluate/               (joblib artifacts)
 predict/save/load)
                   │
                   ▼
          src/prediction/predict_*.py
                   │
                   ▼
             api/main.py (FastAPI)
        POST /predict/{occupancy,pricing,
               restaurant,staff,churn}
```

A standalone Part 1 orchestration layer, `src/pipelines/feature_engineering.py`,
calls the same `src/features/*.py` functions each pipeline uses internally
and additionally persists the result to `data/features/{domain}_features.parquet`
for inspection — it never duplicates feature logic, and pipelines never call
into it (both call into `src/features/*.py`, not each other).

## Folder structure

```
hotelmind-ml/
├── data/
│   ├── raw/              # seed CSVs: room_type_dim, events_holiday_calendar,
│   │                     # restaurant_daily_synthetic, staffing_daily_synthetic
│   ├── processed/        # Phase 3 cleaned dataset (hotel_bookings_clean.parquet)
│   ├── warehouse/        # Phase 3 warehouse loader output (dim_*/fact_booking.parquet)
│   └── features/         # Phase 4 Part 1 output: {domain}_features.parquet
├── notebooks/
├── sql/                  # legacy Postgres-mart queries, unused by current training/prediction code
├── src/
│   ├── config/           # Settings (pydantic-settings), constants
│   ├── database/         # get_connection(), run_query() — only used by the Phase 3 warehouse loader's optional --write-db path
│   ├── features/         # calendar/time-series/preprocessing + domain feature helpers
│   │   └── occupancy_aggregation.py  # derives daily occupancy/revenue from fact_booking
│   ├── models/           # BaseMLModel + per-domain model subclasses
│   │   ├── occupancy/    # xgboost, prophet, lstm (scaffold)
│   │   ├── pricing/       # xgboost
│   │   ├── restaurant/    # xgboost (per meal)
│   │   ├── staffing/      # regression, or_tools_scheduler (scaffold)
│   │   └── churn/         # random_forest, xgboost
│   ├── pipelines/        # BasePipeline + per-domain pipelines
│   │   ├── feature_engineering.py    # Part 1 orchestration -> data/features/*.parquet
│   │   ├── synthetic_data.py         # Restaurant/Staffing synthetic seed generators
│   │   ├── ml_reports.py             # markdown report writers
│   │   └── generate_occupancy_report.py  # Part 2 forecast CSV + metrics JSON
│   ├── training/         # CLI entrypoints: train_*.py
│   ├── prediction/       # predict_*.py — used by both CLI and API
│   ├── evaluation/       # metrics.py, report_writer.py
│   └── utils/            # logging
├── api/                  # FastAPI prediction service
│   ├── main.py
│   ├── schemas.py
│   └── routers/
├── models/               # saved joblib artifacts (*.pkl)
├── reports/
│   ├── features/          # Part 1: feature_dictionary.md, feature_statistics.md, correlation_report.md
│   ├── models/             # Part 7: occupancy_metrics.json, occupancy_forecast.csv, comparison.md, leaderboard.md
│   ├── final_phase4/       # Part 10: phase4_summary.md, training_results.md, api_examples.md, known_limitations.md
│   └── latest_<module>.json  # per-module metrics, written by every training run
├── doc/                  # this documentation
├── tests/
├── requirements.txt
└── .env.example
```

## Model storage convention

One artifact file per algorithm (not one per module), so multiple algorithms
per domain (occupancy: Prophet + XGBoost; churn: Random Forest + XGBoost) can
be compared side by side:

```
models/occupancy_xgboost.pkl   models/occupancy_prophet.pkl
models/pricing_xgboost.pkl
models/restaurant_breakfast.pkl  models/restaurant_lunch.pkl  models/restaurant_dinner.pkl
models/staffing_regression.pkl
models/churn_random_forest.pkl  models/churn_xgboost.pkl
```

`predict_churn` picks whichever of Random Forest / XGBoost scored the higher
ROC-AUC in the last `reports/latest_churn.json`, falling back to XGBoost if
the report is missing.

## Explicitly scaffolded, not implemented

- **`src/models/occupancy/lstm_model.py`** — class skeleton with `NotImplementedError`
  and `# TODO` markers; not wired into the pipeline or API. Prophet + XGBoost
  are the working occupancy models.
- **`src/models/staffing/or_tools_scheduler.py`** — class skeleton with
  `NotImplementedError`; converting `required_staff` counts into actual shift
  assignments via OR-Tools constraint solving is future work.
