# Cleaning Plan

This is a plan only — **no cleaning has been executed**. Target output
location once approved and implemented: `data/processed/hotel_bookings_clean.parquet`,
built from `data/raw/hotel_booking_demand/original/hotel_bookings.csv` (the
recommended canonical source — see `dataset_comparison.md`).

Each rule states: the issue, the exact rule, the row/column impact, and the
rationale.

## 1. Duplicate rows (31,994 rows, 26.8%)

- **Rule:** Deduplicate on the full 32-column tuple; keep first occurrence.
- **Impact:** 119,390 → 87,396 rows.
- **Rationale:** Without a reservation ID, exact-match duplication is the
  only detectable signal. Document in the pipeline log how many were
  dropped so downstream model metrics can be compared with/without this
  step (dedup could remove legitimate coincidental duplicates — treat as a
  documented assumption, not a certainty).
- **Do NOT** apply this rule to `bookings_reduced_columns.csv` if it is ever
  used standalone — 82% of its rows are "duplicates" by construction (only
  10 low-cardinality columns), and deduplicating it would destroy the
  dataset.

## 2. Placeholder "Undefined" categories

- **Rule:** Replace `"Undefined"` with proper NULL in `meal`,
  `market_segment`, `distribution_channel`.
- **Impact:** `meal`: 1,169 rows → NULL; `market_segment`: 2 rows → NULL;
  `distribution_channel`: 5 rows → NULL.
- **Rationale:** These are the dataset's own null-placeholder convention,
  not a genuine 4th/9th/6th category. Leaving them in would teach models an
  artificial category with no real-world meaning.

## 3. Missing `children` (4 rows)

- **Rule:** Impute as `0`.
- **Impact:** 4 rows.
- **Rationale:** Zero is both the mode and the median; negligible volume,
  no benefit to more sophisticated imputation.

## 4. Missing `country` (488 rows, 0.41%)

- **Rule:** Impute as `"UNK"` (new category), not dropped.
- **Impact:** 488 rows retain all other column values.
- **Rationale:** `country` isn't on the critical path for any Phase 4 ML
  module (occupancy/pricing/restaurant/staffing/churn all key off branch,
  date, and stay attributes — not guest nationality). Dropping rows would
  lose real signal in other columns for no benefit.

## 5. Invalid `adr` values

- **Rule (two-step):**
  1. Drop rows where `adr < 0` (impossible — cannot have negative pricing).
  2. Cap / winsorize `adr` at the 99.5th percentile OR drop rows above a
     fixed ceiling (e.g. `adr > 1000`) — **exact threshold needs a decision
     during implementation**, informed by inspecting the actual outlier
     rows (are they real high-value bookings or data errors?).
- **Impact:** Unknown row count until step 2's threshold is chosen; step 1
  affects at minimum the 1 row with adr = -6.38 (likely more below 0, exact
  count needs re-query).
- **Rationale:** `hotel_bookings_cleaned.csv` already demonstrates a
  plausible target range (0.0–211.03 after removing 3,794 rows) — use that
  as a reference point, but re-derive independently rather than trusting an
  unlabeled prior cleaning pass.

## 6. Outlier `adults` (max 55)

- **Rule:** Flag (do not silently drop) rows where `adults > 10`; route to
  a manual-review bucket or cap at a documented ceiling.
- **Impact:** Row count TBD — needs a distribution histogram before
  deciding drop vs. cap vs. keep-and-flag.
- **Rationale:** Could be genuine large group bookings recorded at
  reservation level rather than per-room; blind dropping risks losing valid
  corporate/group data that's actually useful for a "large group" feature.

## 7. `country` code inconsistency (`"CN"` vs ISO-3)

- **Rule:** Map `"CN"` → `"CHN"`.
- **Impact:** 1,279 rows.
- **Rationale:** Every other value in the column is ISO 3166-1 alpha-3;
  normalize so downstream grouping/joins (e.g. against a country-name
  lookup) don't silently split China into two buckets.

## 8. Cancelled bookings

- **Rule:** Do **not** drop cancelled bookings (`is_canceled == 1` /
  `reservation_status == "Canceled"`) from the cleaned dataset. Retain them
  with the status flag intact.
- **Rationale:** Cancellations are directly useful — churn/no-show modeling
  and demand forecasting both need the cancellation signal. Only
  *occupancy*-specific feature engineering (Phase 4, not this cleaning
  pass) should filter `is_canceled == 0` at the point of use, matching the
  pattern already used in the real warehouse (`fact_booking` filters
  `not is_terminal` for `mart_revenue_daily`).

## 9. `reservation_status` vs `is_canceled` consistency check

- **Rule:** Cross-validate: every row with `reservation_status == "Canceled"`
  should have `is_canceled == 1`, and `"Check-Out"`/`"No-Show"` should have
  `is_canceled` consistent with cancellation semantics (No-Show is not the
  same as Canceled — verify no logical contradiction exists between the two
  columns before trusting either downstream).
- **Impact:** Row count TBD pending the cross-tab.
- **Rationale:** Two columns encode overlapping information; a mismatch
  would indicate a labeling bug worth understanding before it's baked into
  a churn or cancellation-prediction feature.

## 10. `meal` type "unknown" handling for feature engineering

- **Rule:** After rule #2 converts `"Undefined"` to NULL, downstream feature
  engineering should treat NULL `meal` as its own explicit "unknown" bucket
  in one-hot encoding (not silently imputed to the mode `"BB"`), so models
  can learn whether "unknown meal plan" itself is predictive.
- **Rationale:** Preserves information rather than fabricating a meal plan
  that wasn't actually selected/recorded.

## Explicitly out of scope for this cleaning pass

- Row-level PII removal — not needed, since the recommended canonical file
  (`hotel_booking_demand`) has no PII columns to begin with. The PII variant
  should simply not be used (see `dataset_comparison.md`).
- Any warehouse-key synthesis (`reservation_id`, `guest_id`, `room_id`,
  `branch_id`) — that is a **loading/mapping** concern
  (`warehouse_mapping.md`), not a data-cleaning concern, and belongs in the
  next milestone.
- Feature engineering (lag features, rolling averages, occupancy
  aggregation) — explicitly deferred to `feature_opportunities.md` /
  Phase 4, per the task's scope boundary.
