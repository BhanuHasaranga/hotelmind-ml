# Correlation Report

## occupancy

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| total_rooms | branch_id | 1.000 |
| occupancy_7day_avg | occupancy_pct_rolling_mean_7 | 0.997 |
| occupancy_pct_rolling_mean_30 | occupancy_30day_avg | 0.996 |
| quarter | month | 0.970 |
| occupancy_pct_lag_1 | occupancy_pct | 0.955 |
| revenue_7day_avg | total_revenue | 0.948 |
| occupancy_pct_lag_1 | occupancy_pct_rolling_mean_7 | 0.941 |
| occupancy_7day_avg | occupancy_pct_lag_1 | 0.937 |
| occupancy_30day_avg | occupancy_7day_avg | 0.930 |
| occupancy_pct_rolling_mean_30 | occupancy_pct_rolling_mean_7 | 0.929 |

## pricing

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| total_rooms | branch_id | 1.000 |
| month | quarter | 0.970 |
| total_revenue | revenue_7day_avg | 0.948 |
| occupancy_7day_avg | occupancy_30day_avg | 0.930 |
| avg_daily_rate | revenue_7day_avg | 0.917 |
| occupancy_pct | occupancy_7day_avg | 0.915 |
| total_revenue | avg_daily_rate | 0.911 |
| occupancy_pct | demand_index | 0.873 |
| occupancy_30day_avg | occupancy_pct | 0.838 |
| occupancy_pct | occupied_rooms | 0.837 |

## restaurant

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| items_sold | total_orders | 1.000 |
| lunch_revenue | lunch_qty_check | 1.000 |
| dinner_revenue | dinner_qty_check | 1.000 |
| breakfast_qty_check | breakfast_revenue | 1.000 |
| breakfast_qty | breakfast_qty_check | 1.000 |
| breakfast_revenue | breakfast_qty | 1.000 |
| lunch_qty | lunch_revenue | 1.000 |
| lunch_qty | lunch_qty_check | 1.000 |
| dinner_qty_check | dinner_qty | 1.000 |
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
| present_employees_rolling_mean_7 | scheduled_employees | 0.883 |
| present_employees_lag_7 | present_employees_rolling_mean_30 | 0.878 |
| scheduled_employees | present_employees_rolling_mean_30 | 0.870 |
| present_employees_lag_7 | present_employees_lag_1 | 0.869 |

## churn

Top 10 absolute pairwise correlations (excluding self-correlation):

| Feature A | Feature B | |corr| |
|---|---|---|
| guest_key | guest_id | 1.000 |
| lifetime_spend | monetary | 1.000 |
| lifetime_bookings | frequency | 1.000 |
| total_nights | lifetime_spend | 0.992 |
| monetary | total_nights | 0.992 |
| monetary | frequency | 0.978 |
| lifetime_spend | lifetime_bookings | 0.978 |
| monetary | lifetime_bookings | 0.978 |
| lifetime_spend | frequency | 0.978 |
| total_nights | lifetime_bookings | 0.975 |
