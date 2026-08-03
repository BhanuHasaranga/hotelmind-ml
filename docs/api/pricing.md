# Dynamic Pricing API

## Purpose

Recommends a room price (`avg_daily_rate`) and expected revenue for a given
branch, room type, and date, using the trained XGBoost pricing model. Takes
current occupancy/revenue context as input rather than looking it up, so the
caller controls the "what-if" scenario being priced.

## Endpoint

```
POST /predict/pricing
Content-Type: application/json
```

## Request body

| Field | Type | Required | Notes |
|---|---|---|---|
| `branch_id` | int | yes | Hotel identifier (echoed back, not otherwise used by the model) |
| `room_type_id` | int | yes | Must exist in `data/raw/room_type_dim.csv` (1=Standard, 2=Deluxe, 3=Suite) — note this endpoint reads the raw seed directly, unlike training which sources room type from `data/warehouse/dim_room_type.parquet` |
| `date` | string (`YYYY-MM-DD`) | yes | Date being priced — drives calendar features (season, weekend, holiday) |
| `current_occupancy_pct` | float | yes | Current occupancy percentage for that date/branch |
| `current_revenue` | float | yes | Current total revenue for that date/branch |
| `revenue_7day_avg` | float | yes | Trailing 7-day revenue average, used to compute `demand_index` |
| `total_rooms` | int | yes | Room capacity, used to scale `expected_revenue` |

```json
{
  "branch_id": 1,
  "room_type_id": 1,
  "date": "2017-09-15",
  "current_occupancy_pct": 55.0,
  "current_revenue": 5000.0,
  "revenue_7day_avg": 4800.0,
  "total_rooms": 200
}
```

## Response body

| Field | Type | Notes |
|---|---|---|
| `branch_id`, `room_type_id`, `date` | | echoed from the request |
| `recommended_price` | float | predicted `avg_daily_rate`, rounded to 2 decimals |
| `expected_revenue` | float | `recommended_price * current_occupancy_pct / 100 * total_rooms`, rounded to 2 decimals |

```json
{
  "branch_id": 1,
  "room_type_id": 1,
  "date": "2017-09-15",
  "recommended_price": 48.92,
  "expected_revenue": 5380.88
}
```

## Validation rules

- Enforced by Pydantic (`api/schemas.py::PricingRequest`): all fields
  required with the stated types.
- `room_type_id` is looked up against `data/raw/room_type_dim.csv` inside
  `recommend_price()` — an unknown `room_type_id` raises `IndexError` at
  lookup time (`.iloc[0]` on an empty match), which is caught by the
  router's generic exception handler and surfaced as a `500`, not a `404`.
  This is a known gap, not by-design behavior — pass one of `{1, 2, 3}`.

## Error responses

| Status | Condition | Body |
|---|---|---|
| `503` | `models/pricing_xgboost.pkl` does not exist yet | `{"detail": "Pricing model not trained yet"}` |
| `500` | Unknown `room_type_id`, or any other exception | `{"detail": "<exception message>"}`, full traceback logged server-side |
| `422` | Request body fails Pydantic validation | FastAPI's standard validation-error body |

## Example curl command

```bash
curl -s -X POST http://127.0.0.1:8000/predict/pricing \
  -H "Content-Type: application/json" \
  -d @demo/sample_requests/pricing.json
```

## Expected output

See `demo/sample_responses/pricing.json` for the exact, real response
captured from this endpoint against the trained model — reproduced above.
