# Restaurant Demand API

## Purpose

Forecasts revenue and estimated order quantity for breakfast, lunch, and
dinner on a given date/branch, using 3 independently trained XGBoost models
(one per meal period).

**Trained on synthetic data** — there is no real restaurant order data
anywhere in this project. See
`reports/final_phase4/known_limitations.md` item 1 before treating any
response from this endpoint as a real business forecast.

## Endpoint

```
POST /predict/restaurant
Content-Type: application/json
```

## Request body

| Field | Type | Required | Notes |
|---|---|---|---|
| `branch_id` | int | yes | Echoed in the response, not otherwise used by the models |
| `date` | string (`YYYY-MM-DD`) | yes | Drives calendar features (season, weekend, holiday) |
| `recent_total_orders_lag_1` | float | yes | Total orders 1 day prior |
| `recent_total_orders_lag_7` | float | yes | Total orders 7 days prior |
| `recent_total_orders_rolling_mean_7` | float | yes | 7-day trailing rolling mean of total orders |
| `avg_item_value` | float | yes | Used to convert each meal's predicted revenue into an estimated quantity |

```json
{
  "branch_id": 1,
  "date": "2017-09-15",
  "recent_total_orders_lag_1": 60,
  "recent_total_orders_lag_7": 65,
  "recent_total_orders_rolling_mean_7": 62,
  "avg_item_value": 14.0
}
```

## Response body

| Field | Type | Notes |
|---|---|---|
| `branch_id`, `date` | | echoed from the request |
| `breakfast` / `lunch` / `dinner` | object | one per meal period |
| `<meal>.expected_revenue` | float | predicted revenue for that meal, rounded to 2 decimals |
| `<meal>.expected_quantity` | float | `expected_revenue / avg_item_value`, rounded to 1 decimal — a simple derived estimate, not a separately-trained quantity model |

```json
{
  "branch_id": 1,
  "date": "2017-09-15",
  "breakfast": {"expected_quantity": 26.6, "expected_revenue": 372.62},
  "lunch": {"expected_quantity": 20.3, "expected_revenue": 283.96},
  "dinner": {"expected_quantity": 40.1, "expected_revenue": 561.65}
}
```

## Validation rules

- Enforced by Pydantic (`api/schemas.py::RestaurantRequest`): all fields
  required with the stated types.
- `avg_item_value == 0` is not rejected but is handled gracefully by
  `forecast_restaurant_demand` (falls back to `expected_quantity = 0.0`
  rather than dividing by zero).

## Error responses

| Status | Condition | Body |
|---|---|---|
| `503` | Any of `models/restaurant_{breakfast,lunch,dinner}.pkl` does not exist yet | `{"detail": "Restaurant models not trained yet"}` |
| `500` | Any other exception during prediction | `{"detail": "<exception message>"}`, full traceback logged server-side |
| `422` | Request body fails Pydantic validation | FastAPI's standard validation-error body |

## Example curl command

```bash
curl -s -X POST http://127.0.0.1:8000/predict/restaurant \
  -H "Content-Type: application/json" \
  -d @demo/sample_requests/restaurant.json
```

## Expected output

See `demo/sample_responses/restaurant.json` for the exact, real response
captured from this endpoint against the trained models — reproduced above.
