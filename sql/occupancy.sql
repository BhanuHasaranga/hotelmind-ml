-- Source: hotelmind_warehouse.mart_occupancy_daily
-- Grain: branch_id x date
-- Params: %(branch_id)s, %(start_date)s, %(end_date)s
SELECT
    branch_id,
    branch_name,
    occupancy_date,
    year,
    month,
    quarter,
    week_of_year,
    day_of_week,
    is_weekend,
    total_rooms,
    occupied_rooms,
    available_rooms,
    occupancy_pct,
    occupancy_7day_avg,
    occupancy_30day_avg,
    occupancy_pct_lag_7d,
    occupancy_pct_lag_365d,
    occupancy_mtd_avg
FROM hotelmind_warehouse.mart_occupancy_daily
WHERE branch_id = %(branch_id)s
  AND occupancy_date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY occupancy_date;
