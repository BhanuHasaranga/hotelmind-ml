# API Examples

All examples below are real request/response pairs, captured by starting
`uvicorn api.main:app` against the models trained in this milestone and
issuing live `curl` requests — not fabricated samples.

Start the server:

```bash
uvicorn api.main:app --reload
```

## Health check

```
GET /health
```
```json
{"status": "ok"}
```

## Occupancy forecast

```
POST /predict/occupancy
Content-Type: application/json

{"branch_id": 1, "horizon_days": 3}
```
```json
{
  "branch_id": 1,
  "forecast": [
    {"date": "2017-09-14", "predicted_occupancy_pct": 50.60, "ci_lower": 23.24, "ci_upper": 77.98, "model_used": "prophet"},
    {"date": "2017-09-15", "predicted_occupancy_pct": 50.26, "ci_lower": 24.13, "ci_upper": 78.08, "model_used": "prophet"},
    {"date": "2017-09-16", "predicted_occupancy_pct": 51.20, "ci_lower": 25.55, "ci_upper": 79.27, "model_used": "prophet"}
  ]
}
```
Note the forecast starts the day after the Prophet model's own training data
ends (2017-09-13), not the real calendar date — see `known_limitations.md`.

## Dynamic pricing

```
POST /predict/pricing
Content-Type: application/json

{
  "branch_id": 1, "room_type_id": 1, "date": "2017-09-15",
  "current_occupancy_pct": 55.0, "current_revenue": 5000.0,
  "revenue_7day_avg": 4800.0, "total_rooms": 200
}
```
```json
{
  "branch_id": 1, "room_type_id": 1, "date": "2017-09-15",
  "recommended_price": 48.92, "expected_revenue": 5380.88
}
```

## Restaurant demand

```
POST /predict/restaurant
Content-Type: application/json

{
  "branch_id": 1, "date": "2017-09-15",
  "recent_total_orders_lag_1": 60, "recent_total_orders_lag_7": 65,
  "recent_total_orders_rolling_mean_7": 62, "avg_item_value": 14.0
}
```
```json
{
  "branch_id": 1, "date": "2017-09-15",
  "breakfast": {"expected_quantity": 26.6, "expected_revenue": 372.62},
  "lunch": {"expected_quantity": 20.3, "expected_revenue": 283.96},
  "dinner": {"expected_quantity": 40.1, "expected_revenue": 561.65}
}
```

## Staff optimization

```
POST /predict/staff
Content-Type: application/json

{
  "branch_id": 1, "department": "Reception", "date": "2017-09-15",
  "scheduled_employees": 5, "present_employees_lag_7": 4,
  "present_employees_rolling_mean_7": 4.5
}
```
```json
{
  "branch_id": 1, "department": "Reception", "date": "2017-09-15",
  "required_staff": 5,
  "confidence_note": "Point estimate from GradientBoostingRegressor; shift assignment still requires OR-Tools scheduling (see src/models/staffing/or_tools_scheduler.py TODO)."
}
```

## Customer churn

```
POST /predict/churn
Content-Type: application/json

{"guest_id": "33462606499639"}
```
```json
{
  "guest_id": "33462606499639",
  "churn_probability": 0.0002,
  "risk_level": "Low",
  "model_used": "xgboost",
  "note": null
}
```

## Error handling

- Model not yet trained → `503 {"detail": "<Module> model(s) not trained yet"}`
- Unknown `guest_id` (churn only) → `404 {"detail": "guest_id <id> not found"}`
- Any other failure → `500 {"detail": "<error message>"}`, with the full
  traceback logged server-side via `src.utils.logging.get_logger`.
