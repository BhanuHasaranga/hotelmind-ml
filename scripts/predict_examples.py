"""Runs all 5 prediction endpoints locally, without needing a running
uvicorn server -- calls the underlying prediction functions directly
(same functions the FastAPI routers call), using the same sample payloads
committed under demo/sample_requests/.

Prints each request/response pair. Exits non-zero if any prediction raises.

Usage:
    python scripts/predict_examples.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.predict_churn import predict_churn  # noqa: E402
from src.prediction.predict_occupancy import forecast_occupancy  # noqa: E402
from src.prediction.predict_pricing import recommend_price  # noqa: E402
from src.prediction.predict_restaurant import forecast_restaurant_demand  # noqa: E402
from src.prediction.predict_staffing import recommend_staffing  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

SAMPLE_REQUESTS_DIR = PROJECT_ROOT / "demo" / "sample_requests"


def _load_request(name: str) -> dict:
    return json.loads((SAMPLE_REQUESTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def run() -> dict[str, object]:
    results: dict[str, object] = {}
    failures: list[str] = []

    checks = [
        ("occupancy", lambda req: forecast_occupancy(branch_id=req["branch_id"], horizon_days=req.get("horizon_days")).to_dict(orient="records")),
        ("pricing", lambda req: recommend_price(**req)),
        ("restaurant", lambda req: forecast_restaurant_demand(**req)),
        ("staff", lambda req: recommend_staffing(
            branch_id=req["branch_id"], department=req["department"], date=req["date"],
            scheduled_employees=req["scheduled_employees"],
            present_employees_lag_7=req["present_employees_lag_7"],
            present_employees_rolling_mean_7=req["present_employees_rolling_mean_7"],
        )),
        ("churn", lambda req: predict_churn(req["guest_id"])),
    ]

    for name, fn in checks:
        request = _load_request(name)
        print(f"\n=== {name} ===")
        print("Request:", json.dumps(request, indent=2))
        try:
            response = fn(request)
            print("Response:", json.dumps(response, indent=2, default=_default))
            results[name] = response
        except Exception as exc:
            logger.exception("Prediction failed for %s", name)
            print(f"FAILED: {exc}")
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} prediction(s) failed: {failures}")
        sys.exit(1)

    print(f"\nAll {len(checks)} prediction endpoints ran successfully.")
    return results


if __name__ == "__main__":
    run()
