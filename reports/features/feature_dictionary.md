# Feature Dictionary

## occupancy

| Column | Dtype |
|---|---|
| branch_id | int64 |
| occupancy_date | datetime64[us] |
| occupied_rooms | int64 |
| total_revenue | float64 |
| avg_daily_rate | float64 |
| branch_name | str |
| total_rooms | int64 |
| occupancy_pct | float64 |
| revenue_7day_avg | float64 |
| occupancy_7day_avg | float64 |
| occupancy_30day_avg | float64 |
| month | int32 |
| quarter | int32 |
| day_of_week | int32 |
| is_weekend | int64 |
| season | str |
| is_public_holiday | int64 |
| is_local_event | int64 |
| event_name | str |
| is_holiday | int64 |
| is_event | int64 |
| occupancy_pct_lag_1 | float64 |
| occupancy_pct_lag_7 | float64 |
| occupancy_pct_lag_30 | float64 |
| occupancy_pct_rolling_mean_7 | float64 |
| occupancy_pct_rolling_mean_30 | float64 |
| occupancy_trend | float64 |
| revenue_trend | float64 |

## pricing

| Column | Dtype |
|---|---|
| branch_id | int64 |
| date | datetime64[us] |
| occupied_rooms | int64 |
| total_revenue | float64 |
| avg_daily_rate | float64 |
| branch_name | str |
| total_rooms | int64 |
| occupancy_pct | float64 |
| revenue_7day_avg | float64 |
| occupancy_7day_avg | float64 |
| occupancy_30day_avg | float64 |
| month | int32 |
| quarter | int32 |
| day_of_week | int32 |
| is_weekend | int64 |
| season | float64 |
| is_public_holiday | int64 |
| is_local_event | int64 |
| event_name | str |
| is_holiday | int64 |
| is_event | int64 |
| demand_index | float64 |
| room_type_name | float64 |
| base_price_multiplier | float64 |

## restaurant

| Column | Dtype |
|---|---|
| branch_id | int64 |
| date | datetime64[us] |
| breakfast_revenue | float64 |
| lunch_revenue | float64 |
| dinner_revenue | float64 |
| total_revenue | float64 |
| avg_item_value | float64 |
| items_sold | int64 |
| total_orders | int64 |
| month | int32 |
| quarter | int32 |
| day_of_week | int32 |
| is_weekend | int64 |
| season | str |
| is_public_holiday | int64 |
| is_local_event | int64 |
| event_name | str |
| is_holiday | int64 |
| is_event | int64 |
| breakfast_qty | float64 |
| breakfast_qty_check | float64 |
| lunch_qty | float64 |
| lunch_qty_check | float64 |
| dinner_qty | float64 |
| dinner_qty_check | float64 |
| total_orders_lag_1 | float64 |
| total_orders_lag_7 | float64 |
| total_orders_rolling_mean_7 | float64 |

## staff

| Column | Dtype |
|---|---|
| branch_id | int64 |
| date | datetime64[us] |
| department_name | float64 |
| department_id | int64 |
| present_employees | int64 |
| scheduled_employees | int64 |
| month | int32 |
| quarter | int32 |
| day_of_week | int32 |
| is_weekend | int64 |
| season | str |
| is_public_holiday | int64 |
| is_local_event | int64 |
| event_name | str |
| is_holiday | int64 |
| is_event | int64 |
| present_employees_lag_1 | float64 |
| present_employees_lag_7 | float64 |
| present_employees_lag_30 | float64 |
| present_employees_rolling_mean_7 | float64 |
| present_employees_rolling_mean_30 | float64 |

## churn

| Column | Dtype |
|---|---|
| guest_key | int64 |
| guest_id | int64 |
| full_name | object |
| nationality | category |
| lifetime_bookings | int64 |
| lifetime_spend | float64 |
| first_stay_date | datetime64[us] |
| last_stay_date | datetime64[us] |
| total_nights | int64 |
| cancellation_ratio | float64 |
| repeat_guest | int64 |
| recency_days | int64 |
| churn | int64 |
| frequency | int64 |
| monetary | float64 |
| avg_spend_per_stay | float64 |
