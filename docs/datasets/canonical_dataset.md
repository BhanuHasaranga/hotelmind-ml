# Canonical Dataset

## Original Kaggle datasets

Seven raw CSV files were profiled during Phase 3 data discovery (full detail
in `reports/data_discovery/dataset_inventory.md`), falling into two
unrelated families:

**Family A — "Hotel Booking Demand"** (all derived from/related to the same
underlying data):
1. `hotel_booking_demand/original/hotel_bookings.csv` — **the canonical file** (119,390 rows, 32 columns)
2. `hotel_booking_demand_indexed/original/hotel_bookings.csv` — identical data + an `index` column
3. `hotel_booking_demand_pii/original/hotel_booking.csv` — identical data + 4 fabricated PII columns (name/email/phone/credit_card), confirmed synthetic (5,000-value ceiling on 119,390 rows)
4. `hotel_bookings_variants/original/bookings_reduced_columns.csv` — 10-column subset, cancellation-feature shaped
5. `hotel_bookings_variants/original/hotel_bookings_cleaned.csv` — 115,596 rows (3,794 fewer — outlier-ADR rows pre-removed), duplicates NOT removed
6. `hotel_bookings_variants/original/hotel_bookings_updated_2024.csv` — same 119,390 rows, relabeled with fabricated Indian cities and a forced `2024` year

**Family B — "Hotel Reservations"** (`hotel_reservations/original/Hotel
Reservations.csv`) — a completely separate, unrelated dataset (36,275 rows,
different schema, `INN#####` booking IDs, 2017–2018 arrivals). Not used.

## Why one dataset became canonical

`hotel_booking_demand/original/hotel_bookings.csv` was selected because:

- It's the **original, unmodified source** — every other Family A file is
  either a derivative (indexed, PII-augmented, column-reduced) or a
  relabeled/partially-cleaned copy of this exact data.
- It has **no PII** — unlike the `_pii` variant, which fabricates guest
  identity fields never actually present in real hotel booking exports.
- It has the **fullest column set** (32 columns) and the **full, unmodified
  row count** (119,390) — the "cleaned" variant already had 3,794 rows
  removed with no documentation of the removal criteria, making it
  unsuitable as a starting point for an auditable cleaning pipeline.
- Its date range (2015-07-01 to 2017-08-31 arrivals) is real; the "2024"
  variant's dates are synthetic relabeling, not real business dates.

Full comparison and reasoning: `reports/data_discovery/dataset_comparison.md`.

## Cleaning pipeline

Implemented in `src/pipelines/data_cleaning.py`, applied in this order:

1. Deduplicate on all 32 columns, keep first occurrence (119,390 → 87,396 rows — 31,994 exact duplicates removed)
2. Replace literal `"Undefined"` placeholder with NULL (`meal`, `market_segment`, `distribution_channel`)
3. Impute missing `children` as 0 (4 rows)
4. Impute missing `country` as `"UNK"` (452 rows)
5. Normalize `"CN"` → `"CHN"` (ISO-3 consistency, 1,093 rows)
6. Drop `adr < 0` (1 row), then winsorize values above the 99.5th percentile (cap=285.00, 435 rows capped)
7. Flag `adults > 10` as `is_adults_outlier` — value preserved, not modified (12 rows)
8. Cross-validate `reservation_status`/`is_canceled` consistency — reporting only, 1,014 mismatches logged, not corrected
9. Trim whitespace on categorical columns
10. Construct `arrival_date` from year/month-name/day
11. Convert to final dtypes (nullable Int64, float64, category, datetime64)

Result: **87,395 cleaned rows**, saved to
`data/processed/hotel_bookings_clean.parquet` (+ `.csv`). Full detail and
approved rules: `reports/data_discovery/cleaning_plan.md`,
`reports/warehouse_loading/loading_summary.md`.

## Known limitations

- No stable per-booking ID exists in the source — "duplicate" is defined as
  an exact match across all 32 columns, which cannot distinguish genuine
  re-exported duplicates from coincidentally identical bookings.
- `adr`'s negative-value and extreme-outlier handling (winsorization at the
  99.5th percentile) is a defensible but not uniquely correct choice — see
  `reports/data_discovery/data_quality.md` for the full reasoning.
- The 1,014 `reservation_status`/`is_canceled` mismatches were deliberately
  left uncorrected (reported, not fixed) since no ground truth exists to
  resolve which field is authoritative.

See `docs/datasets/warehouse.md` for how this cleaned dataset was further
transformed into the star-schema warehouse Phase 4 reads from.
