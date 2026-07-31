-- Source: hotelmind_warehouse.dim_guest joined with an aggregation of fact_booking
-- Grain: one row per guest
-- Params: none (pulls all guests; filter in pandas if needed)
SELECT
    g.guest_key,
    g.guest_id,
    g.full_name,
    g.nationality,
    g.lifetime_bookings,
    g.lifetime_spend,
    g.first_stay_date,
    g.last_stay_date,
    COUNT(b.surrogate_key)                       AS completed_bookings,
    COALESCE(SUM(b.nights), 0)                   AS total_nights,
    COALESCE(AVG(b.total_amount), 0)             AS avg_spend_per_stay
FROM hotelmind_warehouse.dim_guest g
LEFT JOIN hotelmind_warehouse.fact_booking b
       ON b.guest_key = g.guest_key
      AND b.is_completed = TRUE
GROUP BY
    g.guest_key, g.guest_id, g.full_name, g.nationality,
    g.lifetime_bookings, g.lifetime_spend, g.first_stay_date, g.last_stay_date;
