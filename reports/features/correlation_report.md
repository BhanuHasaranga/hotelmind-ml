# Correlation Report

## occupancy

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| branch_id | total_rooms | 1.000 |
| occupancy_7day_avg | occupancy_pct_rolling_mean_7 | 0.997 |
| occupancy_pct_rolling_mean_30 | occupancy_30day_avg | 0.996 |
| month | quarter | 0.970 |
| occupancy_pct | occupancy_pct_lag_1 | 0.955 |
| revenue_7day_avg | total_revenue | 0.948 |
| occupancy_pct_lag_1 | occupancy_pct_rolling_mean_7 | 0.941 |
| occupancy_pct_lag_1 | occupancy_7day_avg | 0.937 |
| occupancy_30day_avg | occupancy_7day_avg | 0.930 |
| occupancy_pct_rolling_mean_7 | occupancy_pct_rolling_mean_30 | 0.929 |

## pricing

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| branch_id | total_rooms | 1.000 |
| quarter | month | 0.970 |
| total_revenue | revenue_7day_avg | 0.948 |
| occupancy_30day_avg | occupancy_7day_avg | 0.930 |
| revenue_7day_avg | avg_daily_rate | 0.917 |
| occupancy_7day_avg | occupancy_pct | 0.915 |
| total_revenue | avg_daily_rate | 0.911 |
| occupancy_pct | demand_index | 0.873 |
| occupancy_30day_avg | occupancy_pct | 0.838 |
| occupied_rooms | occupancy_pct | 0.837 |

## restaurant

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| lunch_revenue | lunch_qty_check | 1.000 |
| total_orders | items_sold | 1.000 |
| dinner_revenue | dinner_qty_check | 1.000 |
| breakfast_revenue | breakfast_qty_check | 1.000 |
| breakfast_qty | breakfast_qty_check | 1.000 |
| breakfast_revenue | breakfast_qty | 1.000 |
| lunch_revenue | lunch_qty | 1.000 |
| lunch_qty_check | lunch_qty | 1.000 |
| dinner_qty | dinner_qty_check | 1.000 |
| dinner_revenue | dinner_qty | 1.000 |

## staff

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| department_name | department_id | 1.000 |
| present_employees | scheduled_employees | 0.981 |
| month | quarter | 0.970 |
| present_employees_rolling_mean_7 | present_employees_rolling_mean_30 | 0.962 |
| present_employees_lag_1 | present_employees_rolling_mean_7 | 0.913 |
| present_employees_rolling_mean_7 | present_employees_lag_7 | 0.909 |
| scheduled_employees | present_employees_rolling_mean_7 | 0.883 |
| present_employees_lag_7 | present_employees_rolling_mean_30 | 0.878 |
| present_employees_rolling_mean_30 | scheduled_employees | 0.870 |
| present_employees_lag_1 | present_employees_lag_7 | 0.869 |

## churn

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| guest_key | guest_id | 1.000 |
| monetary | lifetime_spend | 1.000 |
| lifetime_bookings | frequency | 1.000 |
| total_nights | lifetime_spend | 0.992 |
| total_nights | monetary | 0.992 |
| lifetime_spend | frequency | 0.978 |
| frequency | monetary | 0.978 |
| monetary | lifetime_bookings | 0.978 |
| lifetime_spend | lifetime_bookings | 0.978 |
| frequency | total_nights | 0.975 |
