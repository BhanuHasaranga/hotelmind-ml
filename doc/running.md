# How to Run

## Setup

```bash
cd hotelmind-ml
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in WAREHOUSE_DB_* credentials
```

## Training

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

## Prediction API

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

## Tests

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
