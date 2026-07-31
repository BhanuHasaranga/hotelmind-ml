# Data Quality Report

Scope: `hotel_booking_demand/original/hotel_bookings.csv` (recommended
canonical file — see `dataset_comparison.md`). Issues found in the sibling
variants are noted where they diverge. No cleaning has been applied; this is
observation only.

## 1. Missing values

| Column | Missing | % | Assessment |
|---|---|---|---|
| `company` | 112,593 / 119,390 | 94.3% | Expected — most bookings aren't via a company account. Not a data-quality defect. |
| `agent` | 16,340 / 119,390 | 13.7% | Direct bookings have no agent. Expected. |
| `country` | 488 / 119,390 | 0.41% | Genuine missing values — no placeholder used. |
| `children` | 4 / 119,390 | 0.003% | Negligible, likely blank entry. |

No other columns have missing values.

## 2. Duplicate records

- **31,994 fully duplicate rows (26.8%)** in the canonical file. Because the
  file has no reservation ID, a "duplicate" is defined as an exact match
  across all 32 columns — this could be (a) genuine re-exported duplicate
  reservations, or (b) coincidentally identical bookings (same hotel, same
  dates, same adr, same everything) which is plausible at this row volume
  for a low-cardinality categorical-heavy schema. Cannot be disambiguated
  without a booking ID.
- `bookings_reduced_columns.csv` shows 82.1% duplicate rows — expected, since
  it only keeps 10 low-cardinality columns.
- `hotel_bookings_cleaned.csv` still has 31,822 duplicates despite the
  "cleaned" name — duplicate removal was **not** part of whatever cleaning
  produced that file.

## 3. Invalid / impossible values

- **`adr` (average daily rate): -6.38 to 5400.0.**
  - Negative ADR is impossible for a paid stay — at least one row has
    ADR ≈ -6.38.
  - 5400.0 is 53× the mean (101.83) and is very likely a data-entry error
    or a genuine extreme outlier (single very large multi-room/corporate
    booking) — needs row-level inspection before deciding.
  - `hotel_bookings_cleaned.csv`'s ADR range (0.0–211.03) suggests whoever
    built that file already identified and removed exactly these
    outlier/invalid rows (119,390 → 115,596, a drop of 3,794).
- **`adults`: max 55.** No family/leisure booking plausibly has 55 adults on
  one reservation; likely a data-entry error (missing decimal, or a
  corporate/group booking mis-recorded at the room level rather than
  aggregated). Needs a defined outlier ceiling.
- **`babies`: max 10; `children`: max 10.** Plausible only for a large group
  booking recorded as one row; worth flagging alongside the `adults` outlier
  rather than assuming they're independently wrong.
- **`stays_in_week_nights`: max 50.** A 50-night stay is unusual but not
  impossible (extended corporate stay) — lower priority than `adr`/`adults`.

## 4. Inconsistent / placeholder categories

- **`meal`**: `"Undefined"` used as a literal category value (1,169 rows,
  1.0%) instead of a true null — should be treated as missing during
  cleaning, not as a 5th valid meal-plan category.
- **`market_segment`**: `"Undefined"` (2 rows).
- **`distribution_channel`**: `"Undefined"` (5 rows).
- These three `"Undefined"` strings are the dataset's own placeholder-for-null
  convention and should be normalized to real NULLs, not left as a
  categorical level models would otherwise try to learn from.

## 5. Encoding / format problems

- `reservation_status_date` format is **inconsistent across variants**:
  `YYYY-MM-DD` in the canonical file, `DD-MM-YY` in the indexed variant, and
  full ISO timestamps with microseconds in `hotel_bookings_updated_2024.csv`.
  Any pipeline reading more than one variant must not assume a single date
  format.
- `country` uses ISO 3166-1 alpha-3 codes (`PRT`, `GBR`) except for `"CN"`
  (2-letter, should be `"CHN"`) — a genuine encoding inconsistency inside the
  column, present in every Family A variant (1,279 rows in the canonical
  file use `"CN"`).
- No character-encoding issues detected (file reads cleanly as UTF-8, no
  mojibake in the `country`/`meal`/categorical fields).

## 6. Outliers (numeric columns, beyond `adr`/`adults` above)

- `lead_time`: 0–737 days — 737 days (~2 years ahead) is extreme but
  plausible for a corporate/long-lead booking; mean 104 days, distribution
  is right-skewed as expected for booking lead time. Not flagged as invalid,
  but a candidate for log-transform in feature engineering.
- `days_in_waiting_list`: 0–391 — highly right-skewed (mean 2.32), most
  bookings have 0; the max is plausible for a waitlisted group booking.
- `previous_cancellations`: 0–26 — plausible for a repeat corporate client
  with a poor cancellation history; not flagged.

## 7. Cross-file quality notes

- `hotel_booking_demand_pii`'s fabricated PII (`name`, `email`,
  `phone-number`) cycles over only ~5,000 distinct values across 119,390
  rows — confirms it's synthetic, not real guest data. **Do not treat as a
  usable guest-identity source**, and do not load into any table under any
  circumstance (even fabricated PII columns are unnecessary risk to carry
  forward).
- `hotel_bookings_updated_2024.csv`'s `reservation_status_date` values are
  **generation timestamps** (when the file was fabricated), not real
  business dates — do not use this file for any date-based analysis; it
  would silently inject 2024 dates into what is actually 2015–2017 data.

## Recommended cleaning rules (see `cleaning_plan.md` for execution detail)

1. Drop exact-duplicate rows (32-column match) — flag count but do not drop
   blindly without documenting the ambiguity above.
2. Convert `"Undefined"` in `meal`, `market_segment`, `distribution_channel`
   to true NULL.
3. Convert `country == "CN"` to `"CHN"` for ISO-3 consistency.
4. Filter or cap `adr < 0` (drop) and `adr` above a defined ceiling
   (winsorize or drop, decision needed — see cleaning plan).
5. Filter or cap `adults > <threshold>` (e.g. > 10) as a data-entry-error
   guard; document the chosen threshold.
6. Impute `country` missing (488 rows) as `"Unknown"` rather than dropping —
   country is not critical path for occupancy/pricing/restaurant/staffing
   models.
7. Impute `children` missing (4 rows) as 0 — negligible volume, zero is the
   modal value.
8. Standardize `reservation_status_date` parsing to handle the three
   observed formats defensively if any pipeline ever reads multiple
   variants (recommendation is to read the canonical file only — see
   `warehouse_mapping.md`).
