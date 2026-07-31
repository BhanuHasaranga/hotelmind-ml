import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.routers.churn as churn_router
import api.routers.occupancy as occupancy_router
import api.routers.pricing as pricing_router
import api.routers.restaurant as restaurant_router
import api.routers.staffing as staffing_router
from api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        occupancy_router,
        "forecast_occupancy",
        lambda branch_id, horizon_days=None: pd.DataFrame(
            [
                {
                    "branch_id": branch_id,
                    "occupancy_date": pd.Timestamp("2026-08-01"),
                    "predicted_occupancy_pct": 62.5,
                    "ci_lower": 55.0,
                    "ci_upper": 70.0,
                    "model_used": "prophet",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        pricing_router,
        "recommend_price",
        lambda **kwargs: {
            "branch_id": kwargs["branch_id"],
            "room_type_id": kwargs["room_type_id"],
            "date": kwargs["date"],
            "recommended_price": 120.5,
            "expected_revenue": 4500.0,
        },
    )
    monkeypatch.setattr(
        restaurant_router,
        "forecast_restaurant_demand",
        lambda **kwargs: {
            "branch_id": kwargs["branch_id"],
            "date": kwargs["date"],
            "breakfast": {"expected_quantity": 40.0, "expected_revenue": 400.0},
            "lunch": {"expected_quantity": 60.0, "expected_revenue": 900.0},
            "dinner": {"expected_quantity": 80.0, "expected_revenue": 1600.0},
        },
    )
    monkeypatch.setattr(
        staffing_router,
        "recommend_staffing",
        lambda **kwargs: {
            "branch_id": kwargs["branch_id"],
            "department": kwargs["department"],
            "date": kwargs["date"],
            "required_staff": 5,
            "confidence_note": "stub",
        },
    )
    monkeypatch.setattr(
        churn_router,
        "predict_churn_fn",
        lambda guest_id: {
            "guest_id": guest_id,
            "churn_probability": 0.42,
            "risk_level": "Medium",
            "model_used": "xgboost",
        },
    )
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_occupancy(client):
    resp = client.post("/predict/occupancy", json={"branch_id": 1, "horizon_days": 30})
    assert resp.status_code == 200
    body = resp.json()
    assert body["branch_id"] == 1
    assert body["forecast"][0]["model_used"] == "prophet"


def test_predict_pricing(client):
    resp = client.post(
        "/predict/pricing",
        json={
            "branch_id": 1,
            "room_type_id": 2,
            "date": "2026-08-01",
            "current_occupancy_pct": 65.0,
            "current_revenue": 5000.0,
            "revenue_7day_avg": 4800.0,
            "total_rooms": 100,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["recommended_price"] == 120.5


def test_predict_restaurant(client):
    resp = client.post(
        "/predict/restaurant",
        json={
            "branch_id": 1,
            "date": "2026-08-01",
            "recent_total_orders_lag_1": 100.0,
            "recent_total_orders_lag_7": 95.0,
            "recent_total_orders_rolling_mean_7": 98.0,
            "avg_item_value": 12.5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["breakfast"]["expected_revenue"] == 400.0


def test_predict_staffing(client):
    resp = client.post(
        "/predict/staff",
        json={
            "branch_id": 1,
            "department": "Reception",
            "date": "2026-08-01",
            "scheduled_employees": 6,
            "present_employees_lag_7": 5.0,
            "present_employees_rolling_mean_7": 5.2,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["required_staff"] == 5


def test_predict_churn(client):
    resp = client.post("/predict/churn", json={"guest_id": "g1"})
    assert resp.status_code == 200
    assert resp.json()["risk_level"] == "Medium"
