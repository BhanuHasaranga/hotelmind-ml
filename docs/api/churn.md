# Customer Churn API

## Purpose

Predicts churn probability and risk level for a given guest, using whichever
of Random Forest / XGBoost scored the higher ROC-AUC in the last training
run (`reports/latest_churn.json`), falling back to XGBoost if that report is
missing.

## Endpoint

```
POST /predict/churn
Content-Type: application/json
```

## Request body

| Field | Type | Required | Notes |
|---|---|---|---|
| `guest_id` | string | yes | Must match a `guest_id` value in `data/warehouse/dim_guest.parquet` (compared as a string; the underlying warehouse column is `int64`, cast to `str` before comparison — see `src/prediction/predict_churn.py`) |

```json
{
  "guest_id": "33462606499639"
}
```

## Response body

| Field | Type | Notes |
|---|---|---|
| `guest_id` | string | echoed from the request |
| `churn_probability` | float \| null | 0.0 – 1.0, rounded to 4 decimals; `null` if the guest has zero lifetime bookings (churn isn't meaningful for a guest who never stayed) |
| `risk_level` | string | `"Low"` (&lt;0.3), `"Medium"` (0.3–0.6), `"High"` (&gt;0.6), or `"Unknown"` if `churn_probability` is `null` |
| `model_used` | string | `"random_forest"` or `"xgboost"` |
| `note` | string \| null | explanatory text when `churn_probability` is `null`; otherwise `null` |

```json
{
  "guest_id": "33462606499639",
  "churn_probability": 0.0002,
  "risk_level": "Low",
  "model_used": "xgboost",
  "note": null
}
```

## Validation rules

- Enforced by Pydantic (`api/schemas.py::ChurnRequest`): `guest_id` must be
  a string.
- Guest lookup happens against `data/warehouse/dim_guest.parquet` +
  `fact_booking.parquet` (merged to compute `total_nights`) — no live
  Postgres connection is used.
- `recency_days` for the guest is computed against the dataset's own
  `last_stay_date` maximum, not the real calendar date — see
  `reports/final_phase4/known_limitations.md` item 11.

## Error responses

| Status | Condition | Body |
|---|---|---|
| `503` | Neither `models/churn_random_forest.pkl` nor `models/churn_xgboost.pkl` exists yet | `{"detail": "Churn models not trained yet"}` |
| `404` | `guest_id` not found in `dim_guest.parquet` | `{"detail": "guest_id <id> not found"}` |
| `500` | Any other exception during prediction | `{"detail": "<exception message>"}`, full traceback logged server-side |
| `422` | Request body fails Pydantic validation | FastAPI's standard validation-error body |

## Example curl command

```bash
curl -s -X POST http://127.0.0.1:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d @demo/sample_requests/churn.json
```

## Expected output

See `demo/sample_responses/churn.json` for the exact, real response captured
from this endpoint against the trained model — reproduced above. Note the
near-zero churn probability is not evidence of an especially "loyal" guest —
see `reports/final_phase4/known_limitations.md` item 6 for why churn model
outputs skew toward extreme confidence on this dataset.
