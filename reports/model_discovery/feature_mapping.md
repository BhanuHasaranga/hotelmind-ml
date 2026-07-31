# Feature Mapping — Warehouse Columns → Model Inputs

Column-by-column mapping from the Phase 3 warehouse parquet tables to each
domain's trained-model feature set. Unmapped/not-applicable cells mean that
column isn't used by that domain's model, not that it's missing.

## fact_booking.parquet → Occupancy / Pricing / Churn

| fact_booking column | Occupancy | Pricing | Churn |
|---|:---:|:---:|:---:|
| check_in_date_key / check_out_date_key | ✅ expanded into occupied-room-nights | ✅ (via shared daily base) | |
| branch_key | ✅ → branch_id | ✅ | |
| avg_daily_rate | ✅ → total_revenue | ✅ (target: `avg_daily_rate`) | |
| nights | | | ✅ → `total_nights` (summed per guest_key) |
| reservation_status | ✅ excludes Canceled/No-Show from occupancy | ✅ | ✅ → `cancellation_ratio` |
| guest_key | | | ✅ groups `nights`/cancellations per guest |
| room_key | | ✅ (via dim_room_type join, not directly) | |
| total_amount | NOT MAPPED — `total_revenue` is recomputed from `avg_daily_rate`, not read from this column | | |
| adults, children | NOT MAPPED — no domain currently uses per-booking party size as a feature | | |
| is_adults_outlier | NOT MAPPED | | |

## dim_date.parquet → Occupancy / Pricing / Restaurant / Staffing

| dim_date column | Usage |
|---|---|
| full_date | join key for `occupancy_aggregation.py`'s per-night expansion |
| month, quarter, day_of_week, is_weekend | NOT directly read — `add_calendar_features` recomputes these from the derived date column rather than joining dim_date's precomputed versions, so calendar features stay consistent whether the date came from the warehouse or the synthetic generator |

## dim_hotel.parquet → Occupancy / Pricing / Restaurant / Staffing

| dim_hotel column | Usage |
|---|---|
| hotel_key | → `branch_id` throughout all 4 domains |
| hotel_name | → looked up in `ASSUMED_TOTAL_ROOMS` (Occupancy only) |
| hotel_id (text) | NOT MAPPED |

## dim_room_type.parquet → Pricing

| dim_room_type column | Usage |
|---|---|
| room_type_name | ✅ cross-joined onto every (branch, date) row, categorical feature |
| base_price_multiplier | ✅ numeric feature |
| room_type_key, room_type_id, source_code_examples | NOT MAPPED — pricing doesn't need the room-type surrogate key, only its name/multiplier |

## dim_guest.parquet → Churn

| dim_guest column | Usage |
|---|---|
| guest_key | ✅ join key, and the model's implicit entity ID |
| lifetime_bookings | ✅ → `frequency` |
| lifetime_spend | ✅ → `monetary` |
| first_stay_date | NOT MAPPED — not used as a feature (only `last_stay_date` drives `recency_days`) |
| last_stay_date | ✅ → `recency_days` (via `label_churn`) |
| guest_id, full_name, nationality | NOT MAPPED — `full_name` is blank/scrubbed per Phase 3's warehouse loader (no PII in source data); `nationality` was not selected as a feature this phase |

## Columns with no warehouse source at all (Restaurant, Staffing)

Every column in `restaurant_features.parquet` and `staff_features.parquet`
beyond `branch_id`/`date`/calendar features has **no warehouse source** —
they come entirely from `src/pipelines/synthetic_data.py`, driven by the
derived `occupied_rooms` signal (itself real, from `fact_booking`) but with
fabricated per-meal/per-department constants. See
`reports/model_discovery/feature_quality.md` §1 for the full provenance
breakdown and `reports/final_phase4/known_limitations.md` for why no real
alternative exists in this project.

## Fields the original scaffold expected but that don't exist anywhere

Carried over from Phase 3's `warehouse_mapping.md` gap list — still true in
Phase 4, since no new warehouse tables were added this phase:

- `fact_occupancy_daily` / `mart_occupancy_daily` (room-inventory-based
  occupancy) — Occupancy uses a derived, capacity-assumption-based
  substitute instead (`occupancy_aggregation.py`).
- `mart_revenue_daily` — substituted by the same derived aggregate.
- `mart_restaurant_daily`, `mart_staff_attendance` /
  `fact_staff_attendance` — no substitute exists in real data; substituted
  entirely by synthetic generation.
