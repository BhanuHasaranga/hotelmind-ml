# Warehouse Mapping

Target: `hotelmind_warehouse` schema, as defined in the sibling
`hotelmind-data` dbt project (`dim_guest`, `dim_room`, `dim_branch`/`dim_hotel`,
`dim_date`, `fact_booking`) and the marts consumed by ML
(`mart_occupancy_daily`, `mart_revenue_daily`, `mart_restaurant_daily`,
`mart_staff_daily`). Source: canonical file
`hotel_booking_demand/original/hotel_bookings.csv`.

## Column-level mapping

```
CSV Column                          → Warehouse Table   → Warehouse Column
─────────────────────────────────────────────────────────────────────────
hotel                               → dim_hotel/dim_branch → hotel_name (derive branch_id — see gaps)
is_canceled                         → fact_booking       → is_terminal / reservation_status (derived)
lead_time                           → (none — derived at query time from check_in_date - created_at)
arrival_date_year/month/week/day    → dim_date            → date (composed into a single date), date_key
stays_in_weekend_nights             → fact_booking        → contributes to nights
stays_in_week_nights                → fact_booking        → contributes to nights
adults                              → fact_booking         → adults
children                            → fact_booking         → children
babies                              → (no warehouse column — babies not tracked separately)
meal                                → (no warehouse column — no meal-plan attribute on fact_booking)
country                             → dim_guest            → nationality (proxy only — country ≠ nationality)
market_segment                      → (no warehouse column — fact_booking has no market_segment/channel attribute)
distribution_channel                → (no warehouse column)
is_repeated_guest                   → dim_guest            → derivable from lifetime_bookings > 1 (not a raw flag)
previous_cancellations              → dim_guest            → contributes to lifetime stats (not stored raw)
previous_bookings_not_canceled      → dim_guest            → lifetime_bookings (proxy)
reserved_room_type                  → dim_room             → room_type_name (label match only, no FK)
assigned_room_type                  → dim_room             → room_type_name (label match only, no FK)
booking_changes                     → (no warehouse column)
deposit_type                        → (no warehouse column)
agent                               → (no warehouse column — no agent/booking-channel dimension)
company                             → (no warehouse column)
days_in_waiting_list                → (no warehouse column)
customer_type                       → (no warehouse column — closest analog: dim_guest has no segment field)
adr                                 → fact_booking          → avg_daily_rate
required_car_parking_spaces         → (no warehouse column)
total_of_special_requests           → (no warehouse column)
reservation_status                  → fact_booking          → reservation_status
reservation_status_date             → fact_booking          → updated_at (proxy)
─────────────────────────────────────────────────────────────────────────
(not present in CSV)                → fact_booking          → reservation_id     [GAP]
(not present in CSV)                → fact_booking          → guest_key / guest_id [GAP]
(not present in CSV)                → fact_booking          → room_key / room_id [GAP]
(not present in CSV)                → fact_booking          → branch_key / branch_id [GAP]
(not present in CSV)                → fact_booking          → total_amount, paid_amount, outstanding_amount [GAP]
(not present in CSV)                → dim_guest             → guest_id, full_name, email, phone [GAP — PII variant has fabricated values only]
(not present in CSV)                → fact_occupancy_daily  → total_rooms, occupied_rooms, available_rooms [GAP]
(not present in CSV)                → fact_restaurant_sale  → entire table [GAP — no F&B data in any Kaggle file]
(not present in CSV)                → fact_staff_attendance → entire table [GAP — no staffing data in any Kaggle file]
```

## Warehouse fields that CANNOT be populated from any of these datasets

- **`fact_booking.reservation_id`** — no stable per-row ID in the canonical
  file (row position only). `Hotel Reservations.csv` has `Booking_ID` but is
  a different, unrelated dataset.
- **`fact_booking.room_key` / `dim_room.room_id`** — no numeric room
  identifier, only a letter-coded room *type*, and it doesn't match
  `room_type_dim.csv`'s Standard/Deluxe/Suite naming.
- **`dim_branch` / `dim_hotel` identity** — `hotel` column has only 2 values
  ("City Hotel"/"Resort Hotel"); no branch_id, no city/country per branch (the
  `country` column describes the *guest's* origin, not the hotel's location).
- **`fact_booking.total_amount` / `paid_amount` / `outstanding_amount`** —
  only `adr` (average daily rate) exists; total revenue requires `nights *
  adr`, and there is no payment/outstanding-balance data at all.
- **`dim_guest.guest_id` / real identity** — no guest identifier in any
  Family A file; the PII variant's name/email/phone are fabricated and cycle
  over ~5,000 values, unusable as real identity data.
- **`fact_occupancy_daily`** (total_rooms, occupied_rooms) — Family A has no
  room-inventory or occupancy-count data; occupancy would have to be derived
  by aggregating overlapping stay-date ranges against an assumed total room
  count, which is not present anywhere in these files.
- **`fact_restaurant_sale`**, **`fact_staff_attendance`** — no F&B or
  staffing data exists in any of the seven CSVs. These marts remain entirely
  synthetic/seed-dependent as already documented in `doc/assumptions.md`.

## CSV columns that are unnecessary for the warehouse

- `agent`, `company` — internal Portuguese-hotel booking-agent/company IDs,
  94%+ missing on `company`, no corresponding warehouse dimension.
- `days_in_waiting_list`, `booking_changes`, `deposit_type`,
  `required_car_parking_spaces`, `total_of_special_requests` — no matching
  warehouse column; potentially useful as **ML features** (see
  `feature_opportunities.md`) but not warehouse facts.
- `reserved_room_type` vs `assigned_room_type` — warehouse only tracks the
  room actually booked; the reserved/assigned distinction has no home unless
  a future `room_upgrade_flag` is added.
- All 4 PII columns in `hotel_booking_demand_pii` (`name`, `email`,
  `phone-number`, `credit_card`) — fabricated, not usable, and would be a
  liability to load even if real (PII should never land in `dim_guest` from
  an unverified Kaggle source).
- `index` (in `hotel_booking_demand_indexed`) — an artifact of a prior
  pandas export (`df.to_csv()` without `index=False`), not a business key.

## Summary

Family A's canonical file maps cleanly onto **~9 of `fact_booking`'s ~15
core columns** (dates, adults/children, adr→avg_daily_rate, reservation
status) but supplies **none of the surrogate/foreign keys** the warehouse
star schema requires (`guest_key`, `room_key`, `branch_key`,
`reservation_id`). Loading this data into the real `hotelmind_warehouse`
schema is not a straight import — it requires **synthesizing** a
`reservation_id` (row hash or sequence), fabricating a plausible
`guest_id`/`room_id`/`branch_id` assignment (since none exists), and
deriving `total_amount = nights * adr`. This should be scoped explicitly in
the next milestone rather than assumed as "just load the CSV."
