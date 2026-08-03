# Staff Optimization API

## Purpose

Predicts the required number of present staff for a department/branch/date,
using a single shared GradientBoostingRegressor across all three departments
(Reception, Kitchen, Housekeeping).

**Trained on synthetic data** — there is no real staff-attendance data
anywhere in this project. See
`reports/final_phase4/known_limitations.md` item 1 before treating any
response from this endpoint as a real staffing recommendation.

## Endpoint

```
POST /predict/staff
Content-Type: application/json
```

## Request body

| Field | Type | Required | Notes |
|---|---|---|---|
| `branch_id` | int | yes | Echoed in the response, not otherwise used by the model |
| `department` | string | yes | One of `"Reception"`, `"Kitchen"`, `"Housekeeping"` (`src/config/constants.py::STAFF_DEPARTMENTS`) |
| `date` | string (`YYYY-MM-DD`) | yes | Drives calendar features (season, weekend, holiday) |
| `scheduled_employees` | int | yes | Planned headcount for that shift |
| `present_employees_lag_7` | float | yes | Present-employee count 7 days prior |
| `present_employees_rolling_mean_7` | float | yes | 7-day trailing rolling mean of present employees |

```json
{
  "branch_id": 1,
  "department": "Reception",
  "date": "2017-09-15",
  "scheduled_employees": 5,
  "present_employees_lag_7": 4,
  "present_employees_rolling_mean_7": 4.5
}
```

## Response body

| Field | Type | Notes |
|---|---|---|
| `branch_id`, `department`, `date` | | echoed from the request |
| `required_staff` | int | predicted headcount, rounded to the nearest integer |
| `confidence_note` | string | fixed text noting this is a point estimate and that shift-assignment scheduling (OR-Tools) remains unimplemented |

```json
{
  "branch_id": 1,
  "department": "Reception",
  "date": "2017-09-15",
  "required_staff": 5,
  "confidence_note": "Point estimate from GradientBoostingRegressor; shift assignment still requires OR-Tools scheduling (see src/models/staffing/or_tools_scheduler.py TODO)."
}
```

## Validation rules

- Enforced by Pydantic (`api/schemas.py::StaffingRequest`): all fields
  required with the stated types.
- `department` is **not** validated against the known department list at
  the API layer — an unrecognized department name is passed through to the
  model's categorical encoder, which was fit with
  `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, so
  it will not raise — it will silently encode as `-1` and produce a
  (likely nonsensical) prediction rather than a `4xx` error. Pass one of the
  3 known department names.

## Error responses

| Status | Condition | Body |
|---|---|---|
| `503` | `models/staffing_regression.pkl` does not exist yet | `{"detail": "Staffing model not trained yet"}` |
| `500` | Any other exception during prediction | `{"detail": "<exception message>"}`, full traceback logged server-side |
| `422` | Request body fails Pydantic validation | FastAPI's standard validation-error body |

## Example curl command

```bash
curl -s -X POST http://127.0.0.1:8000/predict/staff \
  -H "Content-Type: application/json" \
  -d @demo/sample_requests/staff.json
```

## Expected output

See `demo/sample_responses/staff.json` for the exact, real response captured
from this endpoint against the trained model — reproduced above.
