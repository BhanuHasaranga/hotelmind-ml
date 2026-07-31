# Training Plan

Approved training rules for every Phase 4 model — mirrors Phase 3's
`cleaning_plan.md` (the ruleset a downstream implementer follows) but for
model training instead of data cleaning. All rules below were actually
executed; see `reports/final_phase4/training_results.md` for the resulting
metrics.

## Common pattern (all 5 domains)

Every pipeline follows `src/pipelines/base_pipeline.py::BasePipeline.run()`:
`load_data → clean → engineer_features → split → train → evaluate → save →
write_report`. No domain reimplements this sequence — only `load_data`,
`clean`, `engineer_features`, `split`, and `build_models` are overridden per
domain.

1. **Occupancy** (`OccupancyPipeline`)
   - `load_data`: `load_daily_occupancy(branch_id)` filtered to the requested date range.
   - Split: chronological (`default_time_series_split`, `test_size=0.2`, no shuffling — appropriate for time series).
   - Models: `OccupancyXGBoostModel` (n_estimators=300, max_depth=5, lr=0.05) trained on `XGB_FEATURE_COLS`; `OccupancyProphetModel` trained on `occupancy_date`/`occupancy_pct` only.
   - Metrics: MAE, RMSE, MAPE (regression).
2. **Pricing** (`PricingPipeline`)
   - `load_data`: same daily occupancy/revenue base, renamed to `date`.
   - `engineer_features`: adds `demand_index`, cross-joins room type from `dim_room_type.parquet`, ordinal-encodes `season`/`room_type_name`, standard-scales numeric columns.
   - Split: chronological, `test_size=0.2`.
   - Model: `PricingXGBoostModel` (n_estimators=300, max_depth=5, lr=0.05), target `avg_daily_rate`.
3. **Restaurant** (`RestaurantPipeline`)
   - `load_data`: reads/generates the synthetic seed via `ensure_restaurant_seed`, filters to branch + date range.
   - `run()` override: loops the 3 meal periods, training one `RestaurantDemandModel` (n_estimators=250, max_depth=4) per meal, target `{meal}_revenue`. No shared model across meals — meal demand patterns differ enough to warrant independence, per the pre-existing model docstring.
4. **Staffing** (`StaffingPipeline`)
   - `load_data`: reads/generates the synthetic seed via `ensure_staffing_seed`, filters to branch + date range.
   - `engineer_features`: lag/rolling features grouped by `department_id` (not globally — each department's history is independent).
   - Split: chronological, `test_size=0.2`.
   - Model: single shared `StaffingRegressionModel` (GradientBoostingRegressor, n_estimators=250, max_depth=4) across all 3 departments, with `department_name` as an encoded categorical feature — one model, not three, since departments share the same schema/grain (unlike restaurant meals).
5. **Churn** (`ChurnPipeline`)
   - `load_data`: `dim_guest.parquet` merged with per-guest `total_nights` aggregated from `fact_booking.parquet`.
   - `engineer_features`: `snapshot_date` anchored to `last_stay_date.max() + 1 day` (not the real clock — see rule 6 below), then `label_churn` + `add_rfm_features`.
   - Split: random (`default_random_split`, `test_size=0.2`, `random_state=42`) — cross-sectional guest data, not a time series.
   - Models: `ChurnRandomForestModel` (300 trees, max_depth=8, `class_weight="balanced"`) and `ChurnXGBoostModel` (300 trees, max_depth=5, dynamic `scale_pos_weight`).
   - Metrics: accuracy, precision, recall, F1, ROC-AUC (classification).

## Cross-cutting rules

6. **Date anchoring**: any snapshot/forecast-origin date must be derived
   from the dataset's own max date, never `dt.date.today()` — the canonical
   dataset only covers 2015-07-01 to 2017-09-13, and anchoring to the real
   clock produces either a degenerate single-class label (churn) or
   nonsensical extrapolated forecasts (occupancy). Applied in
   `churn_pipeline.py`, `predict_churn.py`, `predict_occupancy.py`.
7. **No live Postgres dependency anywhere** — every `load_data()` reads only
   `data/processed/*.parquet` and `data/warehouse/*.parquet`. The pre-existing
   `sql/*.sql` + `run_query()` path is no longer called by any training or
   prediction code.
8. **Synthetic data is deterministic**: `generate_restaurant_daily`/
   `generate_staffing_daily` use `np.random.default_rng(seed=42)` — identical
   input always produces identical synthetic output, so re-running
   `feature_engineering.py` never silently drifts the training data.
9. **LSTM and OR-Tools scheduler are explicitly out of scope for training** —
   `src/models/occupancy/lstm_model.py` and `src/models/staffing/
   or_tools_scheduler.py` remain `NotImplementedError` scaffolds and were not
   touched, per the task's explicit instruction.
10. **Every model was actually fit** — `python -m src.training.train_<module>`
    was run for all 5 domains in this environment; no metric in
    `reports/latest_<module>.json` is a placeholder.
