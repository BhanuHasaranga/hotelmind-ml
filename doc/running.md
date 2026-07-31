# How to Run

## Setup

```bash
cd hotelmind-ml
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

No `.env`/`WAREHOUSE_DB_*` configuration is required for training, prediction,
or the API — every module reads local parquet under `data/processed/` and
`data/warehouse/`. `.env`/`WAREHOUSE_DB_*` is only relevant if you also run
the Phase 3 warehouse loader's optional `python -m src.pipelines.warehouse_loader --write-db` path.

## Feature engineering

Builds the standalone `data/features/*.parquet` datasets used for review, and
generates `reports/features/{feature_dictionary,feature_statistics,
correlation_report}.md`:

```bash
python -m src.pipelines.feature_engineering
```

This also generates the synthetic Restaurant/Staffing seed files under
`data/raw/{restaurant,staffing}_daily_synthetic.csv` on first run (or reuses
them if already present — deterministic, seeded generation, not a live
process). See [doc/assumptions.md](assumptions.md) for what's synthetic and why.

## Training

Each module has a CLI entrypoint. Branch-scoped modules require `--branch-id`;
churn is guest-level and needs none. `--start-date`/`--end-date` default to
`2023-01-01`..today, which does **not** overlap the canonical dataset's real
range (2015-07-01 to 2017-09-13) — pass explicit dates or training runs over
zero rows and fails:

```bash
python -m src.training.train_occupancy --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_pricing --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_restaurant --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_staffing --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_churn

# Occupancy also needs its forecast report generated separately:
python -m src.pipelines.generate_occupancy_report
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
- After training every module, generate the cross-domain comparison:
  ```python
  from src.pipelines.ml_reports import write_comparison_and_leaderboard
  write_comparison_and_leaderboard()
  ```
  Writes `reports/models/comparison.md` (every algorithm's full metrics per task)
  and `reports/models/leaderboard.md` (best model per task by its primary metric —
  MAPE for regression tasks, ROC-AUC for churn).
