# Model Card: Dynamic Pricing

## Purpose

Recommend a room price (`avg_daily_rate`) and expected revenue for a given
branch, room type, and date, based on occupancy/demand/calendar context.

## Input Features

`FEATURE_COLS` (`src/pipelines/pricing_pipeline.py`):

| Feature | Source |
|---|---|
| occupancy_pct, occupancy_7day_avg | `src/features/occupancy_aggregation.py` |
| demand_index | `src/features/pricing_features.py::add_demand_index` — `occupancy_pct * (total_revenue / revenue_7day_avg)` |
| is_weekend, is_holiday, is_event | `src/features/calendar_features.py` |
| base_price_multiplier | `data/warehouse/dim_room_type.parquet` (training) / `data/raw/room_type_dim.csv` (prediction — see known gap below) |
| season, room_type_name | ordinal-encoded categoricals |

Numeric features are standard-scaled (`StandardScaler`) before training.

## Target Variable

`avg_daily_rate` (float) — the average daily rate for a (branch, date),
derived from `fact_booking.avg_daily_rate`.

## Algorithm

**XGBoost** (`XGBRegressor`, n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42). No other algorithm trained for this domain.

## Training Dataset

`data/features/pricing_features.parquet` — 6,420 rows (1,605 daily rows ×
4 room types via cross-join with `dim_room_type.parquet`). Chronological
80/20 train/test split.

## Evaluation Metrics

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| XGBoost | 25.06 | 32.51 | 20.2% |

Source: `reports/latest_pricing.json`.

## Strengths

- MAPE of 20.2% is the best (lowest) among all regression tasks in this
  phase — pricing has a stronger, more stable signal than occupancy or the
  synthetic domains.
- Trained entirely on real, derived data (no synthetic component).

## Limitations

- `demand_index` is a proxy, not a direct measurement — no real demand
  signal exists in the source dataset.
- Room-type cross-join means the model sees the same date's context
  repeated 3× (once per room type) with only `room_type_name`/
  `base_price_multiplier` varying — the model may under-weight true
  room-type-driven price variation if that signal is weak relative to
  date/occupancy effects.

## Known Assumptions

- `base_price_multiplier` (1.0 Standard / 1.4 Deluxe / 2.0 Suite) is itself
  a synthetic seed value from Phase 3, not a measured price differential.
- **Known inconsistency**: training reads room type from `data/warehouse/
  dim_room_type.parquet`, but the prediction endpoint
  (`predict_pricing.py`) reads from `data/raw/room_type_dim.csv` directly.
  Both currently contain the same 3 room types with identical values, so
  predictions are unaffected today, but this is a latent inconsistency —
  see `docs/api/pricing.md` and `reports/final_release/phase4_release_report.md`.

## Future Improvements

- Point the prediction path at the warehouse dimension for consistency with
  training.
- Add a `404`/`422` response for unknown `room_type_id` instead of the
  current unhandled `IndexError` → `500`.
- Explore whether `expected_revenue`'s formula (`price * occupancy_pct/100 *
  total_rooms`) should itself be model-driven rather than a fixed formula.
