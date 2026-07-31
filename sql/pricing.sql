-- Source: hotelmind_warehouse.mart_occupancy_daily joined with mart_revenue_daily
-- Grain: branch_id x date
-- Params: %(branch_id)s, %(start_date)s, %(end_date)s
SELECT
    o.branch_id,
    o.branch_name,
    o.occupancy_date AS date,
    o.year,
    o.month,
    o.quarter,
    o.week_of_year,
    o.day_of_week,
    o.is_weekend,
    o.total_rooms,
    o.occupancy_pct,
    o.occupancy_7day_avg,
    o.occupancy_30day_avg,
    r.avg_daily_rate,
    r.room_revenue,
    r.total_revenue,
    r.revpar,
    r.revenue_7day_avg,
    r.revenue_mtd
FROM hotelmind_warehouse.mart_occupancy_daily o
JOIN hotelmind_warehouse.mart_revenue_daily r
  ON r.branch_id = o.branch_id
 AND r.date = o.occupancy_date
WHERE o.branch_id = %(branch_id)s
  AND o.occupancy_date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY o.occupancy_date;
