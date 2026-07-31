# Architecture

Phase 4 of HotelMind AI builds five ML modules — occupancy forecasting,
dynamic pricing, restaurant demand, staff optimization, and customer churn —
reading **only** from the Phase 3 warehouse marts (`mart_occupancy_daily`,
`mart_revenue_daily`, `mart_restaurant_daily`, `mart_staff_daily`) and
dimension/fact tables (`dim_guest`, `fact_booking`) in the
`hotelmind_warehouse` Postgres schema produced by `hotelmind-data`.

Out of scope for this phase: LLMs, LangChain, RAG, MLflow, Kafka, AWS,
Terraform, Docker changes, monitoring. Those belong to later phases.

## Pipeline shape

Every module follows the same sequence:

```
Load (sql/*.sql via src/database) -> Clean -> Feature Engineer -> Train -> Evaluate -> Save -> Predict
```

This is implemented once as `src/pipelines/base_pipeline.py::BasePipeline`
and subclassed per domain. Every trained model — regardless of algorithm —
implements the same `BaseMLModel` interface (`src/models/base.py`): `train`,
`evaluate`, `predict`, `save`, `load`. This is what lets five different
domains (time series, regression, classification, Prophet, XGBoost,
scikit-learn) share one pipeline shape without duplicating boilerplate.

```
                    ┌─────────────┐
 sql/*.sql  ──────▶ │  Postgres   │
 (warehouse marts)  │  Warehouse  │
                    └──────┬──────┘
                           │ src/database/query.run_query()
                           ▼
                 ┌───────────────────┐
                 │   BasePipeline     │  load -> clean -> engineer_features
                 │  (per-domain impl) │  -> split -> train -> evaluate -> save
                 └─────────┬──────────┘
                           │
              ┌────────────┼─────────────┐
              ▼            ▼             ▼
        BaseMLModel   reports/*.json   models/*.pkl
      (train/evaluate/                 (joblib artifacts)
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

## Folder structure

```
hotelmind-ml/
├── data/
│   ├── raw/            # synthetic seed CSVs (room_type_dim, events_holiday_calendar)
│   ├── processed/
│   └── features/
├── notebooks/
├── sql/                 # all SQL lives here — never inline in Python
├── src/
│   ├── config/          # Settings (pydantic-settings), constants
│   ├── database/        # get_connection(), run_query()
│   ├── features/        # calendar/time-series/preprocessing + domain feature helpers
│   ├── models/           # BaseMLModel + per-domain model subclasses
│   │   ├── occupancy/    # xgboost, prophet, lstm (scaffold)
│   │   ├── pricing/       # xgboost
│   │   ├── restaurant/    # xgboost (per meal)
│   │   ├── staffing/      # regression, or_tools_scheduler (scaffold)
│   │   └── churn/         # random_forest, xgboost
│   ├── pipelines/        # BasePipeline + per-domain pipelines
│   ├── training/         # CLI entrypoints: train_*.py
│   ├── prediction/       # predict_*.py — used by both CLI and API
│   ├── evaluation/       # metrics.py, report_writer.py
│   └── utils/            # logging
├── api/                  # FastAPI prediction service
│   ├── main.py
│   ├── schemas.py
│   └── routers/
├── models/               # saved joblib artifacts (*.pkl)
├── reports/              # evaluation metric reports (JSON)
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
