-- Source: hotelmind_warehouse.mart_staff_daily
-- Grain: branch_id x department_id x date
-- Params: %(branch_id)s, %(start_date)s, %(end_date)s
SELECT
    branch_id,
    branch_name,
    department_id,
    department_name,
    date,
    year,
    month,
    quarter,
    week_of_year,
    day_of_week,
    is_weekend,
    scheduled_employees,
    present_employees,
    late_employees,
    absent_employees,
    total_scheduled_hours,
    total_actual_hours,
    avg_hours_per_employee,
    total_active_employees,
    attendance_rate_pct,
    hours_utilisation_pct
FROM hotelmind_warehouse.mart_staff_daily
WHERE branch_id = %(branch_id)s
  AND date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY department_id, date;
