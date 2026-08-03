import json

import pandas as pd

from src.config.constants import churn_probability_to_risk_level
from src.config.settings import settings
from src.features.churn_features import add_rfm_features, label_churn
from src.mlops.registry.model_registry import ModelRegistry
from src.pipelines.churn_pipeline import FEATURE_COLS


def _best_model_key() -> str:
    """Pick whichever of random_forest/xgboost scored higher ROC-AUC in the
    last evaluation report, falling back to xgboost if the report or one
    model's metrics are missing.
    """
    report_path = settings.reports_dir_path / "latest_churn.json"
    if not report_path.exists():
        return "xgboost"
    try:
        payload = json.loads(report_path.read_text())
        metrics = payload["metrics"]
        rf_auc = metrics.get("random_forest", {}).get("roc_auc", -1)
        xgb_auc = metrics.get("xgboost", {}).get("roc_auc", -1)
        return "random_forest" if rf_auc > xgb_auc else "xgboost"
    except (KeyError, json.JSONDecodeError):
        return "xgboost"


def predict_churn(guest_id: str) -> dict:
    model_key = _best_model_key()
    model = ModelRegistry().load_production(f"churn_{model_key}")

    dim_guest = pd.read_parquet(settings.data_warehouse_dir_path / "dim_guest.parquet")
    fact_booking = pd.read_parquet(settings.data_warehouse_dir_path / "fact_booking.parquet")
    total_nights = fact_booking.groupby("guest_key")["nights"].sum().rename("total_nights")
    guests = dim_guest.merge(total_nights, left_on="guest_key", right_index=True, how="left")
    guests["total_nights"] = guests["total_nights"].fillna(0)

    guest_row = guests[guests["guest_id"].astype(str) == str(guest_id)]
    if guest_row.empty:
        raise ValueError(f"guest_id {guest_id} not found")

    # Anchor to the dataset's own date range, matching training (see
    # src/pipelines/churn_pipeline.py) -- the source data only covers
    # 2015-2017, so recency_days against the real system clock would be
    # wildly out-of-distribution for the trained model.
    snapshot_date = (guests["last_stay_date"].max() + pd.Timedelta(days=1)).date()
    labeled = label_churn(guest_row, snapshot_date=snapshot_date)
    if labeled.empty:
        # Guest has zero lifetime bookings -- churn is not meaningful yet.
        return {
            "guest_id": guest_id,
            "churn_probability": None,
            "risk_level": "Unknown",
            "model_used": model_key,
            "note": "Guest has no completed bookings; churn is not applicable.",
        }
    featured = add_rfm_features(labeled)

    probability = float(model.predict_proba(featured[FEATURE_COLS])[0])
    return {
        "guest_id": guest_id,
        "churn_probability": round(probability, 4),
        "risk_level": churn_probability_to_risk_level(probability),
        "model_used": model_key,
    }
