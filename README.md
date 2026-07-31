# HotelMind ML — Phase 4 (Machine Learning)

Phase 4 of the HotelMind AI portfolio project. Builds five ML modules —
occupancy forecasting, dynamic pricing, restaurant demand, staff optimization,
and customer churn — reading **only** from the Phase 3 warehouse marts
(`mart_occupancy_daily`, `mart_revenue_daily`, `mart_restaurant_daily`,
`mart_staff_daily`) and dimension/fact tables (`dim_guest`, `fact_booking`) in
the `hotelmind_warehouse` Postgres schema produced by `hotelmind-data`.

Out of scope for this phase: LLMs, LangChain, RAG, MLflow, Kafka, AWS,
Terraform, Docker changes, monitoring. Those belong to later phases.

## Architecture

```
Load (sql/*.sql via src/database) -> Clean -> Feature Engineer -> Train -> Evaluate -> Save -> Predict
```

Every module follows this same pipeline, implemented once as
`src/pipelines/base_pipeline.py::BasePipeline` and subclassed per domain.
Every trained model — regardless of algorithm — implements the same
`BaseMLModel` interface (`src/models/base.py`): `train`, `evaluate`,
`predict`, `save`, `load`. This is what lets five different domains (time
series, regression, classification, Prophet, XGBoost, scikit-learn) share one
pipeline shape without duplicating boilerplate.

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
├── tests/
├── requirements.txt
└── .env.example
```

## How to run

```bash
cd hotelmind-ml
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in WAREHOUSE_DB_* credentials
```

### Training

Each module has a CLI entrypoint. Branch-scoped modules require `--branch-id`;
churn is guest-level and needs none:

```bash
python -m src.training.train_occupancy --branch-id 1
python -m src.training.train_pricing --branch-id 1
python -m src.training.train_restaurant --branch-id 1
python -m src.training.train_staffing --branch-id 1
python -m src.training.train_churn
```

Each run writes:
- Model artifact(s) to `models/` (e.g. `occupancy_xgboost.pkl`, `occupancy_prophet.pkl`)
- A metrics report to `reports/latest_<module>.json` (plus a timestamped copy)

### Prediction API

```bash
uvicorn api.main:app --reload
```

Endpoints (all `POST`, JSON body — see `api/schemas.py` for exact fields):

| Endpoint              | Module      |
|-----------------------|-------------|
| `/predict/occupancy`  | Occupancy forecasting |
| `/predict/pricing`    | Dynamic pricing |
| `/predict/restaurant` | Restaurant demand |
| `/predict/staff`      | Staff optimization |
| `/predict/churn`      | Customer churn |

### Tests

```bash
pytest tests/ -v
```

All unit tests run fully offline against synthetic fixtures — no warehouse
connection required. API tests monkeypatch the `predict_*` functions used by
each router.

## Evaluation

- **Regression** (occupancy, pricing, restaurant, staffing): MAE, RMSE, MAPE — `src/evaluation/metrics.py::regression_metrics`.
- **Classification** (churn): accuracy, precision, recall, F1, ROC-AUC — `src/evaluation/metrics.py::classification_metrics`.
- All metrics are written per-module to `reports/latest_<module>.json` by `src/evaluation/report_writer.py`.

## Assumptions & Data Gaps

The Phase 3 warehouse only exposes branch-level daily aggregates (and a guest
dimension), so a few inputs the spec calls for don't exist as real warehouse
columns. Each is clearly synthetic/derived, not measured:

1. **Room type** (`data/raw/room_type_dim.csv`) — synthetic seed
   (Standard / Deluxe / Suite + a `base_price_multiplier`), since no
   room-type dimension exists in the warehouse. Cross-joined onto pricing
   rows in `src/features/pricing_features.py::add_room_types`.
2. **Events / holiday calendar** (`data/raw/events_holiday_calendar.csv`) —
   synthetic seed: real US public holidays plus a fabricated set of
   recurring "local events," generated once and stored as a static CSV
   (2023–2026). Joined by date in `src/features/calendar_features.py`.
3. **Pricing "demand" input** — no direct demand measure exists, so
   `demand_index = occupancy_pct * (total_revenue / revenue_7day_avg)` is
   used as a proxy in `src/features/pricing_features.py::add_demand_index`
   ("demand is elevated when both occupancy and revenue momentum are high").
4. **Restaurant "expected quantity"** — `mart_restaurant_daily` has
   per-meal *revenue* (breakfast/lunch/dinner) and a single day-level
   `items_sold`, but no per-meal item count. Quantity is approximated as
   `items_sold * (meal_revenue / total_revenue)`, cross-checked against
   `meal_revenue / avg_item_value` (see
   `src/features/restaurant_features.py::derive_meal_quantities`; >20%
   divergence between the two estimates is logged).
5. **Churn label** — no churn flag exists in the warehouse. Defined as a
   recency cutoff: `churn = 1` if `(snapshot_date - last_stay_date).days >
   CHURN_WINDOW_DAYS` (default 180, configurable via `.env`) **and**
   `lifetime_bookings >= 1` (guests who never stayed are excluded — churn
   isn't meaningful for them). See `src/features/churn_features.py::label_churn`.
6. **Churn risk-level thresholds** — business-defined defaults, not fit
   from data: probability `< 0.3` → Low, `0.3–0.6` → Medium, `> 0.6` → High
   (`src/config/constants.py::churn_probability_to_risk_level`).
7. **Season mapping** — standard meteorological seasons by month
   (`src/config/constants.py::MONTH_TO_SEASON`), not hotel-specific
   high/low season definitions (which would require booking-price
   elasticity analysis out of scope for this phase).

## Explicitly scaffolded, not implemented

- **`src/models/occupancy/lstm_model.py`** — class skeleton with `NotImplementedError`
  and `# TODO` markers; not wired into the pipeline or API. Prophet + XGBoost
  are the working occupancy models.
- **`src/models/staffing/or_tools_scheduler.py`** — class skeleton with
  `NotImplementedError`; converting `required_staff` counts into actual shift
  assignments via OR-Tools constraint solving is future work.

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
