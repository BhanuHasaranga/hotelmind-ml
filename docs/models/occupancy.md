# Model Card: Occupancy Forecasting

## Purpose

Forecast daily room occupancy percentage per hotel branch, with a
confidence interval, to support capacity planning and demand-driven pricing
decisions.

## Input Features

`XGB_FEATURE_COLS` (`src/pipelines/occupancy_pipeline.py`):

| Feature | Source |
|---|---|
| month, quarter, day_of_week, is_weekend | `src/features/calendar_features.py` |
| is_holiday, is_event | joined from `data/raw/events_holiday_calendar.csv` |
| total_rooms | fixed capacity assumption (`occupancy_aggregation.py::ASSUMED_TOTAL_ROOMS`) |
| occupancy_pct_lag_1 / _7 / _30 | `src/features/time_series_features.py::add_lag_features` |
| occupancy_pct_rolling_mean_7 / _30 | same module, `add_rolling_features` |

Prophet uses only `occupancy_date` (as `ds`) and the target as `y` — it does
not consume the engineered feature columns above.

## Target Variable

`occupancy_pct` (float, 0–100) — the percentage of `total_rooms` occupied
on a given (branch, date), derived from `fact_booking`'s check-in/check-out
ranges (see `src/features/occupancy_aggregation.py`).

## Algorithm

Two models trained and compared:

- **XGBoost** (`XGBRegressor`, n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
- **Prophet** (`interval_width=0.95`, `daily_seasonality=False`) — the only
  model exposed by the prediction API, since it natively produces a
  confidence interval

**LSTM** (`src/models/occupancy/lstm_model.py`) remains an intentional
`NotImplementedError` scaffold — not trained, per explicit task scope.

## Training Dataset

`data/features/occupancy_features.parquet` — 1,605 rows (branch 1: 806,
branch 2: 799), date range 2015-07-01 to 2017-09-13. Chronological 80/20
train/test split (`default_time_series_split`, no shuffling).

## Evaluation Metrics

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| XGBoost | 4.77 | 7.73 | 72.6% |
| Prophet | 38.63 | 40.51 | 119.5% |

Source: `reports/latest_occupancy.json`.

## Strengths

- XGBoost's MAE (4.77 percentage points) is low in absolute terms — useful
  for operational planning even though MAPE looks poor.
- Prophet natively provides a calibrated confidence interval, which XGBoost
  does not — valuable for risk-aware capacity decisions even at a higher
  point-forecast error.
- No live database dependency — fully reproducible from parquet.

## Limitations

- High MAPE (72–120%) is driven by days with near-zero `occupancy_pct`,
  where any absolute error becomes a large percentage error — MAE/RMSE are
  more informative for this target than MAPE.
- Only ~2.5 years of history — not enough for Prophet's yearly seasonality
  to help meaningfully (disabled in this configuration).
- `total_rooms` is a fixed assumption (200 for Resort Hotel, 300 for City
  Hotel), not a measured room-inventory count — every `occupancy_pct` value
  is relative to this assumption.

## Known Assumptions

- Forecasts anchor to the day after the model's own training data ends
  (`model.history["ds"].max()`), not the real system clock — forecasting
  from "today" against 2015–2017 training data would extrapolate roughly a
  decade forward and produce meaningless (even negative) values.
- Cancelled/no-show bookings are excluded from occupied-room-night counts.

## Future Improvements

- Replace the fixed `total_rooms` constant with a real room-inventory
  dimension, if one becomes available.
- Implement the LSTM scaffold for sequence-based forecasting once deep
  learning infrastructure is justified.
- Add per-branch Prophet models (currently a single global model regardless
  of `branch_id` in the request) if branch-level forecast divergence proves
  material.
