# Warehouse Transformation

How `data/processed/hotel_bookings_clean.parquet` (87,395 rows) becomes the
star-schema warehouse Phase 4 reads from — `data/warehouse/{dim_date,
dim_hotel,dim_guest,dim_room_type,fact_booking}.parquet`.

## Why a local warehouse, not a live database

No Phase 3 warehouse schema (DDL) exists anywhere in this repository or its
sibling projects at the time this pipeline was built — the real Postgres
`hotelmind_warehouse` schema was expected to come from a separate
`hotelmind-data` project. Rather than block on that dependency,
`src/pipelines/warehouse_loader.py` writes the warehouse **locally as
parquet**, always, as the source of truth — with an optional `--write-db`
flag that attempts a live Postgres load and degrades gracefully (logs
clearly, doesn't crash) if the target schema isn't present. This is what
lets Phase 4 train and predict with zero database dependency.

## Surrogate key strategy (deterministic, not random)

| Key | Strategy |
|---|---|
| `date_id` | `int(YYYYMMDD)` from the calendar date |
| `hotel_id` / `branch_id` | fixed lookup: `{"Resort Hotel": 1, "City Hotel": 2}` — only 2 known values, unknown raises an error rather than silently mapping |
| `room_type_id` | letter-code → tier mapping (A-C→Standard, D-G→Deluxe, H/L→Suite), unmapped codes → explicit "Unmapped" bucket (id=0), never dropped |
| `guest_id` | SHA-256 hash of (country, market_segment, distribution_channel, customer_type, is_repeated_guest, agent, company) — identifies a **guest profile cluster**, not a true individual (the anonymized source has no real guest ID) |
| `reservation_id` | SHA-256 hash of all cleaned-row values + a stable post-sort row index, guarding against two coincidentally-identical bookings colliding |

Every key is a pure function of its inputs — the same row always produces
the same key, across every re-run, with no randomness. Full detail:
`reports/warehouse_loading/key_generation.md`.

## Resulting warehouse tables

| Table | Rows | Grain |
|---|---|---|
| `dim_date` | 806 | one row per distinct date referenced by any booking |
| `dim_hotel` | 2 | Resort Hotel, City Hotel |
| `dim_room_type` | 4 | Standard, Deluxe, Suite, Unmapped |
| `dim_guest` | 5,672 | one row per guest profile cluster |
| `fact_booking` | 87,395 | one row per cleaned booking |

Validated after loading: 0 duplicate primary keys, 0 foreign-key orphans,
0 NULL violations on required fields — see
`reports/warehouse_loading/validation_report.md`.

## Fields that could NOT be populated

Per the original warehouse-loading task scope, these were **not
fabricated** — simply left unpopulated:

- `total_amount` is derived (`nights × avg_daily_rate`); `paid_amount`/
  `outstanding_amount` don't exist anywhere in the source data and are not
  created.
- `fact_restaurant_sale` and `fact_staff_attendance` tables were never
  built — no data in any source CSV supports them. This is exactly why
  Phase 4's Restaurant and Staffing domains needed synthetic data (see
  `docs/datasets/synthetic_data.md`).
- No pre-aggregated daily mart (`mart_occupancy_daily`, `mart_revenue_daily`,
  `mart_restaurant_daily`, `mart_staff_daily`) exists in this warehouse —
  only per-booking grain (`fact_booking`). Phase 4 derives its own daily
  occupancy/revenue aggregate from this raw grain
  (`src/features/occupancy_aggregation.py`).

Full mapping detail: `reports/warehouse_loading/mapping_summary.md`,
`reports/model_discovery/feature_mapping.md`.
