"""Part 9: verifies the full train -> save -> load -> predict -> API
response chain end-to-end, with no live PostgreSQL dependency. Uses tiny
in-memory DataFrames (not the full 87K-row dataset) so the test stays fast.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.models.churn.xgboost_model import ChurnXGBoostModel
from src.models.occupancy.xgboost_model import OccupancyXGBoostModel


def _tiny_occupancy_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "month": rng.integers(1, 13, n),
        "quarter": rng.integers(1, 5, n),
        "day_of_week": rng.integers(0, 7, n),
        "is_weekend": rng.integers(0, 2, n),
        "is_holiday": rng.integers(0, 2, n),
        "is_event": rng.integers(0, 2, n),
        "total_rooms": 200,
        "occupancy_pct_lag_1": rng.uniform(0, 100, n),
        "occupancy_pct_lag_7": rng.uniform(0, 100, n),
        "occupancy_pct_lag_30": rng.uniform(0, 100, n),
        "occupancy_pct_rolling_mean_7": rng.uniform(0, 100, n),
        "occupancy_pct_rolling_mean_30": rng.uniform(0, 100, n),
    }), pd.Series(rng.uniform(0, 100, n))


def test_occupancy_xgboost_train_save_load_predict(tmp_path: Path):
    X, y = _tiny_occupancy_frame()

    model = OccupancyXGBoostModel()
    model.train(X, y)
    metrics = model.evaluate(X, y)
    assert set(metrics) == {"mae", "rmse", "mape"}

    save_path = tmp_path / "occupancy_xgboost.pkl"
    model.save(save_path)
    assert save_path.exists()

    loaded = OccupancyXGBoostModel()
    loaded.load(save_path)
    preds = loaded.predict(X)
    assert len(preds) == len(X)
    assert np.allclose(preds, model.predict(X))


def _tiny_churn_frame(n: int = 60):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "recency_days": rng.integers(0, 400, n),
        "frequency": rng.integers(1, 10, n),
        "monetary": rng.uniform(0, 1000, n),
        "avg_spend_per_stay": rng.uniform(0, 300, n),
        "total_nights": rng.integers(0, 20, n),
    })
    y = pd.Series((X["recency_days"] > 180).astype(int))
    return X, y


def test_churn_xgboost_train_save_load_predict_proba(tmp_path: Path):
    X, y = _tiny_churn_frame()

    model = ChurnXGBoostModel()
    model.train(X, y)
    metrics = model.evaluate(X, y)
    assert set(metrics) >= {"accuracy", "precision", "recall", "f1"}

    save_path = tmp_path / "churn_xgboost.pkl"
    model.save(save_path)

    loaded = ChurnXGBoostModel()
    loaded.load(save_path)
    proba = loaded.predict_proba(X)
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_api_occupancy_endpoint_with_real_lazy_loaded_model(tmp_path, monkeypatch):
    """End-to-end: train a real (tiny) model, save it where predict_occupancy
    expects to find it, then hit the API router directly -- not a mocked
    predict function -- and confirm a real response comes back.
    """
    from src.config.settings import settings
    from src.models.occupancy.prophet_model import OccupancyProphetModel

    monkeypatch.setattr(type(settings), "model_dir_path", property(lambda self: tmp_path))

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    y = pd.Series(50 + 10 * np.sin(np.arange(60) / 5))
    prophet_model = OccupancyProphetModel()
    prophet_model.train(pd.DataFrame({"occupancy_date": dates}), y)
    prophet_model.save(tmp_path / "occupancy_prophet.pkl")

    from api.main import app

    client = TestClient(app)
    resp = client.post("/predict/occupancy", json={"branch_id": 1, "horizon_days": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["branch_id"] == 1
    assert len(body["forecast"]) == 5
    assert body["forecast"][0]["model_used"] == "prophet"
