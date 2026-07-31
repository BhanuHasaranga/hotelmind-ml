# Feature & Model Data Quality Report

Scope: the 5 feature datasets under `data/features/` and the models trained
against them. Unlike Phase 3's `data_quality.md` (which profiled a single
raw file), this report's central concern is **provenance** — how much of
each domain's signal is real (derived from actual bookings) versus
fabricated (synthetic seed data), since that materially changes how each
model's output should be trusted.

## 1. Provenance by domain

| Domain | Provenance | Basis |
|---|---|---|
| Occupancy | **Real, derived** | Aggregated from `fact_booking`'s actual check-in/check-out ranges. Only `total_rooms` (capacity) is an assumption, not measured. |
| Pricing | **Real, derived** | Same daily occupancy/revenue base as Occupancy, plus warehouse room-type dimension (real mapping, built in Phase 3). |
| Restaurant | **Synthetic** | No restaurant order data exists anywhere in this project (not in the raw Kaggle CSVs, not in the warehouse). Values are generated, not measured. |
| Staffing | **Synthetic** | Same as Restaurant — no staff attendance data exists anywhere in this project. |
| Churn | **Real** | Derived directly from `dim_guest.parquet` + `fact_booking.parquet`, both Phase-3-verified warehouse tables. |

Restaurant and Staffing model outputs must **never** be presented as
measuring real demand or real staffing needs — see
`reports/final_phase4/known_limitations.md` item 1 for the full caveat.

## 2. Missing values

No missing values were found in any of the 5 feature datasets after
`handle_missing_values(strategy="median")` was applied inside each domain's
feature-engineering step (median imputation is only applied to lag/rolling
columns, which are legitimately null for each series' first few rows before
enough history accumulates). Upstream, `hotel_bookings_clean.parquet` and
the warehouse parquet tables were already validated null-free on their
required fields in Phase 3 (`reports/warehouse_loading/validation_report.md`).

## 3. Distributional sanity checks

- **`occupancy_pct`**: 0.67% – 92.0%, mean 57.2%, std 22.1% — plausible range,
  never exceeds 100% (`build_daily_occupancy` clips at 100).
- **`churn` label balance**: 3,414 churned (60.2%) / 2,258 not churned
  (39.8%) — reasonably balanced, not the degenerate single-class result that
  occurs if `snapshot_date` is left at its `today()` default against this
  2015–2017 dataset (see item 5 below).
- **`recency_days`**: 1 – 793 days, mean 302.6 — spans both sides of the
  180-day churn threshold, confirming the label isn't trivially constant.
- **Restaurant quantity cross-check**: `derive_meal_quantities`'s two
  independent quantity estimates (revenue-share vs. `revenue/avg_item_value`)
  diverge by well under 1% on sampled rows (e.g. 18.35 vs. 18.21) — expected,
  since `avg_item_value` is a single fixed constant in the synthetic
  generator rather than a per-row measured value, so the two formulas are
  nearly equivalent by construction. This confirms the synthetic generator's
  internal consistency, not real-world quantity accuracy.

## 4. Known date-range mismatch (the single largest "gotcha" in this phase)

The canonical dataset covers **2015-07-01 to 2017-09-13 only**. Several
pre-existing defaults in the scaffolded code assumed a live, ongoing data
feed and anchored on the real system clock (`dt.date.today()`), which this
dataset cannot support:

- Training CLI `--start-date`/`--end-date` defaults (`2023-01-01`..today)
  don't overlap the real data at all — training silently runs over zero rows
  unless explicit dates are passed (fixed in `doc/running.md`, not in code
  defaults, since the flags themselves are correct).
- `label_churn`'s default `snapshot_date=today()` would make every guest's
  recency exceed the 180-day window, collapsing `churn` to a single class —
  fixed by anchoring to the guest data's own `last_stay_date` maximum
  (`src/pipelines/churn_pipeline.py`, `src/prediction/predict_churn.py`).
- `predict_occupancy.py::forecast_occupancy`'s original `today()` anchor
  produced **negative occupancy percentages** when forecasting ~9 years past
  the Prophet model's training window — fixed by anchoring to the model's
  own training data end (`model.history["ds"].max()`).

Both fixes are unit-tested (`tests/test_pipelines.py`,
`tests/test_integration_train_predict.py`) and documented in
`doc/assumptions.md` items 11–12.

## 5. Model-level quality signal: churn's near-perfect scores

Random Forest and XGBoost both score ~1.0 ROC-AUC on the churn task. This is
**not** a data leakage bug to fix — `churn` is defined deterministically as
`recency_days > 180`, and `recency_days` is itself a model input feature, so
the classifier is essentially learning a threshold on its own input. This
was a user-approved label definition carried over from an earlier milestone.
Treat these metrics as confirmation the pipeline is wired correctly, not as
evidence of real-world predictive strength — see
`reports/final_phase4/training_results.md`.
