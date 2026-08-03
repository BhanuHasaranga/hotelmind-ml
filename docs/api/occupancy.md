# Occupancy Forecast API

## Purpose

Forecasts daily room occupancy percentage for a hotel branch, `horizon_days`
ahead, using the trained Prophet model. Returns a point forecast plus a 95%
confidence interval per day.

## Endpoint

```
POST /predict/occupancy
Content-Type: application/json
```

## Request body

| Field | Type | Required | Notes |
|---|---|---|---|
| `branch_id` | int | yes | Hotel identifier — `1` = Resort Hotel, `2` = City Hotel (see `data/warehouse/dim_hotel.parquet`) |
| `horizon_days` | int \| null | no | Number of days to forecast; defaults to `settings.FORECAST_HORIZON_DAYS` (30) if omitted |

```json
{
  "branch_id": 1,
  "horizon_days": 5
}
```

## Response body

| Field | Type | Notes |
|---|---|---|
| `branch_id` | int | echoed from the request |
| `forecast` | array of objects | one entry per forecast day |
| `forecast[].date` | string (`YYYY-MM-DD`) | forecast date |
| `forecast[].predicted_occupancy_pct` | float | point forecast |
| `forecast[].ci_lower` / `ci_upper` | float | 95% confidence interval bounds (from Prophet, `interval_width=0.95`) |
| `forecast[].model_used` | string | always `"prophet"` — Prophet is the only model this endpoint serves (XGBoost is trained for comparison in `reports/models/comparison.md`, not exposed here) |

```json
{
  "branch_id": 1,
  "forecast": [
    {
      "date": "2017-04-05",
      "predicted_occupancy_pct": 50.60,
      "ci_lower": 24.71,
      "ci_upper": 78.28,
      "model_used": "prophet"
    }
  ]
}
```

## Validation rules

- Enforced by Pydantic (`api/schemas.py::OccupancyRequest`): `branch_id` must
  be an integer; `horizon_days`, if provided, must be an integer.
- No branch-existence check is performed at the API layer — an unknown
  `branch_id` is echoed back in the response without affecting the forecast
  (the forecast itself is branch-agnostic once the model is loaded — the
  trained Prophet model is per-deployment, not per-branch, in the current
  implementation).

## Error responses

| Status | Condition | Body |
|---|---|---|
| `503` | `models/occupancy_prophet.pkl` does not exist yet | `{"detail": "Occupancy model not trained yet"}` |
| `500` | Any other exception during prediction | `{"detail": "<exception message>"}`, full traceback logged server-side |
| `422` | Request body fails Pydantic validation (e.g. `branch_id` is a string) | FastAPI's standard validation-error body |

## Example curl command

```bash
curl -s -X POST http://127.0.0.1:8000/predict/occupancy \
  -H "Content-Type: application/json" \
  -d @demo/sample_requests/occupancy.json
```

## Expected output

See `demo/sample_responses/occupancy.json` for the exact, real response
captured from this endpoint against the trained model — reproduced above in
abbreviated form. The forecast start date anchors to the day after the
Prophet model's own training data ends (2017-09-13 in this project's
canonical dataset), not the real calendar date — see
`reports/final_phase4/known_limitations.md` item 12.
