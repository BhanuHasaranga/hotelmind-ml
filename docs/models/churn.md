# Model Card: Customer Churn Prediction

## Purpose

Predict the probability that a guest has churned (i.e. is unlikely to
return), and bucket that probability into a Low/Medium/High risk level, to
support targeted retention efforts.

## Input Features

`FEATURE_COLS` (`src/pipelines/churn_pipeline.py`):

| Feature | Source |
|---|---|
| recency_days | days since `last_stay_date`, computed by `label_churn` |
| frequency | = `lifetime_bookings` (`src/features/churn_features.py::add_rfm_features`) |
| monetary | = `lifetime_spend` |
| avg_spend_per_stay | `lifetime_spend / lifetime_bookings` |
| total_nights | aggregated from `fact_booking.nights` per `guest_key` |

## Target Variable

`churn` (int, 0/1) — `1` if `recency_days > CHURN_WINDOW_DAYS` (180, from
`src/config/settings.py`), else `0`. Guests with `lifetime_bookings == 0`
are excluded (churn isn't meaningful for a guest who never stayed).

## Algorithm

Two models trained and compared:

- **Random Forest** (`RandomForestClassifier`, n_estimators=300, max_depth=8, `class_weight="balanced"`, random_state=42)
- **XGBoost** (`XGBClassifier`, n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, dynamic `scale_pos_weight`, random_state=42)

The prediction API automatically selects whichever scored the higher
ROC-AUC in the last training run (`reports/latest_churn.json`), falling
back to XGBoost if that report is missing.

## Training Dataset

`data/features/churn_features.parquet` — 5,672 rows (one per guest profile
cluster in `dim_guest.parquet`). Random 80/20 train/test split
(`random_state=42`) — cross-sectional data, not time series.

## Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 |

Source: `reports/latest_churn.json`.

## Strengths

- Fast to train (well under a second for both algorithms on this dataset
  size).
- `class_weight="balanced"` (Random Forest) and dynamic `scale_pos_weight`
  (XGBoost) correctly account for the 60/40 churn/non-churn class split.
- Label balance is reasonable (3,414 churned / 2,258 not churned) —
  confirms the date-anchoring fix (see Known Assumptions) is working as
  intended, avoiding the degenerate single-class outcome the default
  `snapshot_date=today()` would produce on this dataset.

## Limitations

- **Near-perfect scores are a property of the label definition, not
  evidence of strong real-world predictive power.** `churn` is defined
  deterministically as `recency_days > 180`, and `recency_days` is itself
  one of the model's 5 input features — the classifier is essentially
  learning a threshold on its own input. This was a user-approved label
  definition carried over from an earlier milestone, not a modeling choice
  made to inflate metrics.
- `guest_key` is a booking-profile hash (country, market segment,
  distribution channel, customer type, repeat-guest flag, agent, company),
  **not a true individual guest identity** — the anonymized source dataset
  has no real guest ID, so many distinct real guests may collide into the
  same `guest_key`. See `reports/warehouse_loading/mapping_summary.md`.

## Known Assumptions

- `snapshot_date` (used to compute `recency_days`) is anchored to the
  dataset's own `last_stay_date` maximum plus one day, not the real system
  clock — the canonical dataset only covers 2015–2017, so anchoring to
  "today" would make every guest's recency exceed 180 days and collapse the
  label to a single class.
- `CHURN_WINDOW_DAYS=180` and the Low/Medium/High risk thresholds
  (`<0.3`/`0.3–0.6`/`>0.6`) are business-defined defaults, not fit from
  data (`src/config/constants.py`).

## Future Improvements

- If real guest identity data ever becomes available, retrain against true
  per-guest history instead of profile-cluster aggregates.
- Consider whether a non-deterministic churn label (e.g. probabilistic
  survival modeling) would produce more actionable risk scores than a hard
  180-day cutoff.
