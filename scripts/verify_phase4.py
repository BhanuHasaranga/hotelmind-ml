"""Verifies Phase 4 is complete and reproducible, without retraining or
modifying anything. Checks:

  1. Feature datasets exist       (data/features/*.parquet)
  2. Models exist                 (models/*.pkl)
  3. Reports exist                (reports/latest_*.json, reports/models/*)
  4. API imports                  (api.main can be imported without error)
  5. Prediction works             (each predict_* function runs against demo/sample_requests/)
  6. Required folders exist       (docs/, demo/, scripts/, reports/final_phase4/, reports/model_discovery/)

Prints a PASS/FAIL line per check and a final summary. Exits non-zero if
any check fails.

Usage:
    python scripts/verify_phase4.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CheckResult = tuple[str, bool, str]


def check_feature_datasets() -> CheckResult:
    expected = ["occupancy", "pricing", "restaurant", "staff", "churn"]
    missing = [
        name for name in expected
        if not (PROJECT_ROOT / "data" / "features" / f"{name}_features.parquet").exists()
    ]
    ok = not missing
    detail = "all 5 present" if ok else f"missing: {missing}"
    return ("Feature datasets exist", ok, detail)


def check_models() -> CheckResult:
    expected = [
        "occupancy_prophet.pkl", "occupancy_xgboost.pkl",
        "pricing_xgboost.pkl",
        "restaurant_breakfast.pkl", "restaurant_lunch.pkl", "restaurant_dinner.pkl",
        "staffing_regression.pkl",
        "churn_random_forest.pkl", "churn_xgboost.pkl",
    ]
    missing = [name for name in expected if not (PROJECT_ROOT / "models" / name).exists()]
    ok = not missing
    detail = f"all {len(expected)} present" if ok else f"missing: {missing}"
    return ("Trained models exist", ok, detail)


def check_reports() -> CheckResult:
    expected_json = [f"latest_{m}.json" for m in ["occupancy", "pricing", "restaurant", "staffing", "churn"]]
    missing = [name for name in expected_json if not (PROJECT_ROOT / "reports" / name).exists()]

    expected_models_reports = ["comparison.md", "leaderboard.md", "occupancy_forecast.csv", "occupancy_metrics.json"]
    missing += [
        f"models/{name}" for name in expected_models_reports
        if not (PROJECT_ROOT / "reports" / "models" / name).exists()
    ]

    ok = not missing
    detail = "all present" if ok else f"missing: {missing}"
    return ("Reports exist", ok, detail)


def check_api_imports() -> CheckResult:
    try:
        import api.main  # noqa: F401
        return ("API imports", True, "api.main imported successfully")
    except Exception as exc:
        return ("API imports", False, f"import failed: {exc}")


def check_prediction_works() -> CheckResult:
    sample_requests_dir = PROJECT_ROOT / "demo" / "sample_requests"
    try:
        from src.prediction.predict_churn import predict_churn
        from src.prediction.predict_occupancy import forecast_occupancy
        from src.prediction.predict_pricing import recommend_price
        from src.prediction.predict_restaurant import forecast_restaurant_demand
        from src.prediction.predict_staffing import recommend_staffing

        occ_req = json.loads((sample_requests_dir / "occupancy.json").read_text())
        forecast_occupancy(branch_id=occ_req["branch_id"], horizon_days=occ_req.get("horizon_days"))

        prc_req = json.loads((sample_requests_dir / "pricing.json").read_text())
        recommend_price(**prc_req)

        rest_req = json.loads((sample_requests_dir / "restaurant.json").read_text())
        forecast_restaurant_demand(**rest_req)

        staff_req = json.loads((sample_requests_dir / "staff.json").read_text())
        recommend_staffing(
            branch_id=staff_req["branch_id"], department=staff_req["department"], date=staff_req["date"],
            scheduled_employees=staff_req["scheduled_employees"],
            present_employees_lag_7=staff_req["present_employees_lag_7"],
            present_employees_rolling_mean_7=staff_req["present_employees_rolling_mean_7"],
        )

        churn_req = json.loads((sample_requests_dir / "churn.json").read_text())
        predict_churn(churn_req["guest_id"])

        return ("Prediction works", True, "all 5 predict functions ran successfully")
    except Exception as exc:
        return ("Prediction works", False, f"prediction failed: {exc}")


def check_required_folders() -> CheckResult:
    expected = [
        "docs/api", "docs/architecture", "docs/models", "docs/datasets", "docs/demo",
        "demo/sample_requests", "demo/sample_responses",
        "scripts",
        "reports/final_phase4", "reports/model_discovery", "reports/models", "reports/features",
    ]
    missing = [name for name in expected if not (PROJECT_ROOT / name).is_dir()]
    ok = not missing
    detail = "all present" if ok else f"missing: {missing}"
    return ("Required folders exist", ok, detail)


def run() -> bool:
    checks = [
        check_feature_datasets,
        check_models,
        check_reports,
        check_api_imports,
        check_prediction_works,
        check_required_folders,
    ]

    results = [check() for check in checks]

    print("=== Phase 4 Verification ===\n")
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} - {detail}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    all_ok = passed == total

    print(f"\n{passed}/{total} checks passed.")
    print("OVERALL: PASS" if all_ok else "OVERALL: FAIL")

    return all_ok


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
