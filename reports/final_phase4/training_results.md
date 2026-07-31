# Training Results

All models below were trained via `python -m src.training.train_<module>` in
this environment against `data/processed/`/`data/warehouse/` parquet (branch
1 / Resort Hotel, date range 2015-07-01 to 2017-09-13 — the canonical
dataset's full real range). Metrics are the actual evaluation output, not
placeholders. Source: `reports/latest_<module>.json`.

## Occupancy (Prophet + XGBoost)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| xgboost | 4.77 | 7.73 | 72.6% |
| prophet | 38.63 | 40.51 | 119.5% |

XGBoost outperforms Prophet on this dataset (short 2-year history, two
seasons of data only — not enough for Prophet's yearly-seasonality strength
to help, and it's disabled here since `daily_seasonality=False` and the
series is too short for reliable yearly decomposition). High MAPE on both is
expected: `occupancy_pct` includes days near 0%, where any percentage error
is amplified. LSTM remains a documented scaffold (`src/models/occupancy/
lstm_model.py`, `NotImplementedError`) — not trained, as specified.

## Pricing (XGBoost)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| xgboost | 25.06 | 32.51 | 20.2% |

Predicts `avg_daily_rate` from occupancy/demand/calendar/room-type features.

## Restaurant Demand (3 independent XGBoost models, synthetic data)

| Meal | MAE | RMSE | MAPE |
|---|---|---|---|
| breakfast | 112.79 | 150.91 | 61.4% |
| lunch | 88.47 | 116.50 | 65.2% |
| dinner | 161.14 | 212.86 | 60.8% |

**Trained on synthetic data** — see `known_limitations.md`. Metrics reflect
how well each model fits the synthetic generator's own occupancy-driven
signal plus injected noise, not real restaurant demand.

## Staff Optimization (GradientBoostingRegressor, synthetic data)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| regression | 0.73 | 0.86 | 6.5% |

**Trained on synthetic data** — see `known_limitations.md`. Low MAPE reflects
the synthetic generator's low intrinsic noise (headcounts are small integers
with limited variance), not real-world predictive strength. OR-Tools shift
scheduling remains a documented scaffold (`src/models/staffing/
or_tools_scheduler.py`, `NotImplementedError`) — not implemented, as specified.

## Customer Churn (Random Forest + XGBoost)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| random_forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xgboost | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 |

Near-perfect scores are **expected, not a red flag to "fix"**: `churn` is
defined deterministically as `recency_days > 180`, and `recency_days` is
itself one of the model's input features — the classifier is essentially
learning a threshold on its own input. This is a property of the label
definition (documented and user-approved in `doc/assumptions.md`), not
information leakage from an external source. See `known_limitations.md`.

## Best model per task

See `reports/models/leaderboard.md` for the generated leaderboard. Summary:

| Task | Best model | Primary metric |
|---|---|---|
| Occupancy | XGBoost | MAPE 72.6% (vs. Prophet 119.5%) |
| Pricing | XGBoost | MAPE 20.2% (only model trained) |
| Restaurant | Dinner model | MAPE 60.8% (lowest of the 3 meals) |
| Staffing | Regression | MAPE 6.5% (only model trained) |
| Churn | Random Forest | ROC-AUC 1.000 (ties XGBoost; RF's F1 is marginally higher) |
