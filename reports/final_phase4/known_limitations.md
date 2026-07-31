# Known Limitations

Consolidated list of every assumption, fabrication, and design gap
introduced or surfaced in Phase 4. See `doc/assumptions.md` for the
per-item technical detail; this file is the single prominent index.

## Data provenance

1. **Restaurant and Staffing training data is entirely synthetic.** No
   restaurant order data or staff attendance data exists anywhere in this
   project — not in the raw Kaggle hotel-bookings CSVs, not in the Phase 3
   warehouse. `src/pipelines/synthetic_data.py` generates deterministic,
   seeded daily values driven by real derived occupancy (not pure random
   noise), stored under `data/raw/{restaurant,staffing}_daily_synthetic.csv`.
   **Every Restaurant and Staffing prediction is illustrative, not a real
   business forecast.** Do not present Restaurant/Staffing model outputs to
   stakeholders as measuring real demand.
2. **Occupancy/revenue is derived, not measured.** No pre-aggregated daily
   mart exists in the warehouse — `src/features/occupancy_aggregation.py`
   expands each booking's check-in→check-out range into occupied-room-nights
   from `fact_booking`. This is arithmetically sound but depends on:
3. **Room capacity is a fixed illustrative constant.** No room-inventory
   count exists in the source dataset. `ASSUMED_TOTAL_ROOMS = {"Resort
   Hotel": 200, "City Hotel": 300}` in `occupancy_aggregation.py` is not a
   measured value — every `occupancy_pct` figure in this project is relative
   to this assumption.
4. **Room type mapping is approximate.** The source dataset's room codes are
   single letters (A, B, C, ...) with no published mapping to Standard/
   Deluxe/Suite; this mapping was fixed in the Phase 3 warehouse loading
   milestone and is reused here as-is.

## Date-anchoring (dataset range vs. real clock)

5. **The canonical dataset only covers 2015-07-01 to 2017-09-13.** Several
   places in the original scaffold defaulted to anchoring on `dt.date.
   today()`, which — given this dataset — would silently produce nonsensical
   results:
   - **Occupancy forecast**: `predict_occupancy.py::forecast_occupancy` was
     fixed to anchor the forecast start date to the trained Prophet model's
     own training data end (`model.history["ds"].max()`), not the real
     clock. Before this fix, forecasting from "today" (an ~9-year gap)
     produced negative occupancy percentages.
   - **Churn label/prediction**: `label_churn`'s default `snapshot_date=
     today()` would make every guest's recency exceed `CHURN_WINDOW_DAYS`
     and collapse the label to a single class (unlearnable). Both
     `churn_pipeline.py` and `predict_churn.py` were fixed to anchor
     `snapshot_date` to one day after the guest data's own `last_stay_date`
     maximum.
   - **Training CLI default dates**: `train_occupancy.py`/`train_pricing.py`/
     `train_restaurant.py`/`train_staffing.py` all default `--start-date
     2023-01-01 --end-date <today>`, which does not overlap the real data
     range at all. Pass explicit dates (`--start-date 2015-07-01 --end-date
     2017-09-13`) — this is documented in `doc/running.md` but not enforced
     in code (out of scope for this milestone to change CLI defaults, since
     the flags themselves are correct and pre-existing).

## Model behavior

6. **Churn models score ~100% on every metric.** This is a property of the
   label definition (`churn = recency_days > 180`, and `recency_days` is
   itself a model input), not information leakage from an unrelated source.
   The label was user-approved in an earlier milestone. Treat churn model
   metrics as a sanity check that the pipeline works correctly, not as
   evidence of real-world predictive power on a label defined this way.
7. **Occupancy MAPE is high (72–120%) despite a low MAE (4.8–38.6 points).**
   `occupancy_pct` has many days near 0%, where any absolute error becomes a
   large percentage error. MAE/RMSE are the more informative metrics for
   this target.
8. **Only ~2.5 years of history.** Not enough data for Prophet's yearly
   seasonality to help (disabled: `daily_seasonality=False`, and the series
   is too short for reliable yearly decomposition) — XGBoost outperforms
   Prophet here specifically because of this data-volume constraint, not
   because Prophet is unsuitable for occupancy forecasting in general.

## Explicitly out of scope (matches original task instructions)

9. **LSTM** (`src/models/occupancy/lstm_model.py`) and **OR-Tools shift
   scheduling** (`src/models/staffing/or_tools_scheduler.py`) remain
   `NotImplementedError` scaffolds with `# TODO` markers, exactly as
   instructed ("LSTM remains scaffold only with TODO", "Leave OR-Tools
   scheduler as TODO only"). Not built out in this milestone.
10. **Phase 5** (LangChain, LLMs, RAG, MLflow, Kafka, Docker changes,
    Terraform, AWS, monitoring, streaming, digital twin, executive
    dashboard) — not started, not touched.

## Environment

11. **Python 3.14** is the only interpreter available in this environment.
    Prophet's pip package plus its `cmdstanpy`/`cmdstan` C++ backend were
    successfully installed and verified end-to-end (a real `Prophet().
    fit()`/`.predict()` cycle ran) — no environment blocker for any model in
    this milestone.
