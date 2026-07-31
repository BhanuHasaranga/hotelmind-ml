# Assumptions & Data Gaps

The Phase 3 warehouse only exposes branch-level daily aggregates (and a guest
dimension), so a few inputs the spec calls for don't exist as real warehouse
columns. Each is clearly synthetic/derived, not measured:

1. **Room type** (`data/raw/room_type_dim.csv`) — synthetic seed
   (Standard / Deluxe / Suite + a `base_price_multiplier`), since no
   room-type dimension exists in the warehouse. Cross-joined onto pricing
   rows in `src/features/pricing_features.py::add_room_types`.
2. **Events / holiday calendar** (`data/raw/events_holiday_calendar.csv`) —
   synthetic seed: real US public holidays plus a fabricated set of
   recurring "local events," generated once and stored as a static CSV
   (2023–2026). Joined by date in `src/features/calendar_features.py`.
3. **Pricing "demand" input** — no direct demand measure exists, so
   `demand_index = occupancy_pct * (total_revenue / revenue_7day_avg)` is
   used as a proxy in `src/features/pricing_features.py::add_demand_index`
   ("demand is elevated when both occupancy and revenue momentum are high").
4. **Restaurant "expected quantity"** — `mart_restaurant_daily` has
   per-meal *revenue* (breakfast/lunch/dinner) and a single day-level
   `items_sold`, but no per-meal item count. Quantity is approximated as
   `items_sold * (meal_revenue / total_revenue)`, cross-checked against
   `meal_revenue / avg_item_value` (see
   `src/features/restaurant_features.py::derive_meal_quantities`; >20%
   divergence between the two estimates is logged).
5. **Churn label** — no churn flag exists in the warehouse. Defined as a
   recency cutoff: `churn = 1` if `(snapshot_date - last_stay_date).days >
   CHURN_WINDOW_DAYS` (default 180, configurable via `.env`) **and**
   `lifetime_bookings >= 1` (guests who never stayed are excluded — churn
   isn't meaningful for them). See `src/features/churn_features.py::label_churn`.
6. **Churn risk-level thresholds** — business-defined defaults, not fit
   from data: probability `< 0.3` → Low, `0.3–0.6` → Medium, `> 0.6` → High
   (`src/config/constants.py::churn_probability_to_risk_level`).
7. **Season mapping** — standard meteorological seasons by month
   (`src/config/constants.py::MONTH_TO_SEASON`), not hotel-specific
   high/low season definitions (which would require booking-price
   elasticity analysis out of scope for this phase).
