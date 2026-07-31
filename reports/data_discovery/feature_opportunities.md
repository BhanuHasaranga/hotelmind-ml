# Feature Opportunities (Not Implemented)

Future ML features identifiable from the recommended canonical dataset
(`hotel_booking_demand/original/hotel_bookings.csv`) plus the existing
warehouse marts. No feature engineering code has been written — this is a
forward-looking catalogue only, per task scope.

Modules: **Occupancy**, **Pricing**, **Restaurant**, **Staffing**, **Churn**
(matching the five Phase 4 modules already scaffolded in
`hotelmind-ml/src/models/`).

| Feature | Derivation | Occupancy | Pricing | Restaurant | Staffing | Churn |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `booking_lead_time` | `lead_time` (already a raw column) | ✅ | ✅ | | | |
| `stay_duration` | `stays_in_weekend_nights + stays_in_week_nights` | ✅ | ✅ | ✅ | | |
| `is_weekend_stay` | any weekend night in the stay range | ✅ | ✅ | | | |
| `holiday_flag` | join `events_holiday_calendar.csv` on arrival date | ✅ | ✅ | ✅ | ✅ | |
| `local_event_flag` | join `events_holiday_calendar.csv` `is_local_event` | ✅ | ✅ | ✅ | ✅ | |
| `season` | month → meteorological season (already defined: `MONTH_TO_SEASON`) | ✅ | ✅ | ✅ | ✅ | |
| `arrival_month` / `arrival_day_of_week` | already raw or derivable from `dim_date` | ✅ | ✅ | ✅ | ✅ | |
| `occupancy_rate` | requires room-inventory denominator (not in Family A — warehouse-level only, via `fact_occupancy_daily`) | ✅ | ✅ | | ✅ | |
| `avg_daily_rate` (ADR) | `adr` (already a raw column, after cleaning) | | ✅ | | | |
| `revenue_per_stay` | `adr * total_nights` | | ✅ | | | ✅ |
| `party_size` | `adults + children + babies` | | | ✅ | ✅ | |
| `is_family_booking` | `children > 0 or babies > 0` (matches `hotel_bookings_cleaned.csv`'s existing `is_family`) | | | ✅ | | ✅ |
| `meal_plan_type` | `meal` (already raw, post-cleaning) | | | ✅ | | |
| `repeat_guest_flag` | `is_repeated_guest` (already raw) | | | | | ✅ |
| `customer_lifetime_value` | requires cross-booking aggregation by guest — **not computable from Family A** (no guest ID); warehouse-level only via `dim_guest.lifetime_spend` | | | | | ✅ |
| `cancellation_rate` | `previous_cancellations / (previous_cancellations + previous_bookings_not_canceled)` | | | | | ✅ |
| `booking_channel` | `market_segment` / `distribution_channel` (raw, post-cleaning) | | ✅ | | | ✅ |
| `deposit_risk_flag` | `deposit_type == "No Deposit"` (higher no-show/cancel risk) | | | | | ✅ |
| `special_requests_count` | `total_of_special_requests` (already raw) | | | | ✅ | ✅ |
| `waiting_list_days` | `days_in_waiting_list` (already raw) | ✅ | | | | |
| `room_type_upgrade_flag` | `assigned_room_type != reserved_room_type` | | ✅ | | | |
| `lead_time_bucket` | binned `lead_time` (last-minute / 1-4wk / 1-3mo / 3mo+) | ✅ | ✅ | | | |
| `booking_changes_count` | `booking_changes` (already raw) | | | | | ✅ |
| `nights_lag_7d` / `occupancy_pct_lag_7d` | already implemented at warehouse level (`mart_occupancy_daily`) | ✅ | ✅ | | | |
| `revenue_7day_avg` | already implemented at warehouse level (`mart_revenue_daily`) | | ✅ | ✅ | | |

## Notes on feasibility gaps

- **`occupancy_rate` and `customer_lifetime_value` cannot be derived from
  Family A alone** — both require a denominator (total rooms) or an entity
  key (guest ID) that doesn't exist in the flat Kaggle export. These remain
  warehouse-mart-level features (`fact_occupancy_daily`, `dim_guest`), as
  already implemented in the `hotelmind-data` dbt project and consumed by
  the existing `sql/occupancy.sql` / `sql/guest.sql` queries.
- **Restaurant module features are the weakest fit** for Family A — the
  dataset has no F&B transaction data at all; the only restaurant-adjacent
  signal is `meal` (meal plan selected) and `party_size`, both weak proxies
  for actual F&B demand. Confirms the existing assumption in
  `doc/assumptions.md` (#4) that restaurant quantity is a derived estimate,
  not measured.
- **Staffing module features are similarly indirect** — nothing in Family A
  maps to actual staff scheduling; `party_size`, `special_requests_count`,
  and `holiday_flag` are workload *proxies* at best, consistent with staffing
  already being sourced from `fact_staff_attendance`, not booking data.
