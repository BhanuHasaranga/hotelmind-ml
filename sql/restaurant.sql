-- Source: hotelmind_warehouse.mart_restaurant_daily
-- Grain: branch_id x date
-- Params: %(branch_id)s, %(start_date)s, %(end_date)s
SELECT
    branch_id,
    branch_name,
    date,
    year,
    month,
    quarter,
    week_of_year,
    day_of_week,
    is_weekend,
    total_orders,
    items_sold,
    total_revenue,
    avg_item_value,
    breakfast_revenue,
    lunch_revenue,
    dinner_revenue,
    avg_order_value,
    revenue_7day_avg,
    orders_7day_avg
FROM hotelmind_warehouse.mart_restaurant_daily
WHERE branch_id = %(branch_id)s
  AND date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY date;
