import json

import pandas as pd

from src.config.constants import churn_probability_to_risk_level
from src.config.settings import settings
from src.database.query import run_query
from src.features.churn_features import add_rfm_features, label_churn
from src.models.churn.random_forest_model import ChurnRandomForestModel
from src.models.churn.xgboost_model import ChurnXGBoostModel
from src.pipelines.churn_pipeline import FEATURE_COLS

_MODEL_CLASSES = {
    "random_forest": ChurnRandomForestModel,
    "xgboost": ChurnXGBoostModel,
}


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
    model = _MODEL_CLASSES[model_key]()
    model.load(settings.model_dir_path / f"churn_{model_key}.pkl")

    guests = run_query(settings.sql_dir_path / "guest.sql")
    guest_row = guests[guests["guest_id"] == guest_id]
    if guest_row.empty:
        raise ValueError(f"guest_id {guest_id} not found")

    labeled = label_churn(guest_row)
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
