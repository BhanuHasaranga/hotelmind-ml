"""Direct tests for genai/insights/service.py's `_collect_churn_findings`
and `_collect_review_findings` helpers, which the main
test_insights_service.py suite stubs out via the autouse fixture. These
exercise the real implementations against fakes for their dependencies
(predict_churn, dim_guest.parquet, reviews.service)."""

from __future__ import annotations

import pandas as pd
import pytest

from genai.insights import service as insights_service
from genai.reviews.service import ReviewsNotAnalyzedError
from src.config.settings import settings


def test_collect_churn_findings_samples_guests_when_none_given(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_WAREHOUSE_DIR", str(tmp_path))
    dim_guest = pd.DataFrame({"guest_id": [1, 2, 3]})
    dim_guest.to_parquet(tmp_path / "dim_guest.parquet", index=False)

    monkeypatch.setattr(
        insights_service,
        "predict_churn",
        lambda guest_id: {"guest_id": guest_id, "churn_probability": 0.8, "total_nights": 20},
    )
    findings = insights_service._collect_churn_findings()
    assert len(findings) == 3
    assert all(f.category == "churn" for f in findings)


def test_collect_churn_findings_missing_warehouse_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_WAREHOUSE_DIR", str(tmp_path / "does_not_exist"))
    findings = insights_service._collect_churn_findings()
    assert findings == []


def test_collect_churn_findings_skips_failed_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_WAREHOUSE_DIR", str(tmp_path))
    dim_guest = pd.DataFrame({"guest_id": [1, 2]})
    dim_guest.to_parquet(tmp_path / "dim_guest.parquet", index=False)

    def _predict(guest_id):
        if guest_id == "1":
            raise ValueError("guest not found")
        return {"guest_id": guest_id, "churn_probability": 0.9, "total_nights": 30}

    monkeypatch.setattr(insights_service, "predict_churn", _predict)
    findings = insights_service._collect_churn_findings()
    assert len(findings) == 1


def test_collect_churn_findings_with_explicit_guest_ids(monkeypatch):
    monkeypatch.setattr(
        insights_service,
        "predict_churn",
        lambda guest_id: {"guest_id": guest_id, "churn_probability": 0.7, "total_nights": 5},
    )
    findings = insights_service._collect_churn_findings(guest_ids=["g1", "g2"])
    assert len(findings) == 2


def test_collect_review_findings_returns_findings_when_analyzed(monkeypatch):
    monkeypatch.setattr(insights_service, "get_summary", lambda: {"csat_by_hotel": [{"hotel_id": "H1", "csat": 40.0}]})
    monkeypatch.setattr(insights_service, "get_complaints", lambda: {"complaints": []})
    monkeypatch.setattr(insights_service, "get_trends", lambda grain="weekly": {"trend_by_hotel": {}})

    findings = insights_service._collect_review_findings()
    assert len(findings) == 1
    assert findings[0].category == "guest_experience"


def test_collect_review_findings_handles_not_analyzed(monkeypatch):
    def _raise():
        raise ReviewsNotAnalyzedError("not analyzed")

    monkeypatch.setattr(insights_service, "get_summary", _raise)
    findings = insights_service._collect_review_findings()
    assert findings == []


def test_generate_findings_integrates_all_sources(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=8).astype(str)
    revenue_df = pd.DataFrame(
        {
            "branch_id": [1] * 8,
            "date": dates,
            "total_revenue": [1000] * 8,
            "revenue_7day_avg": [1000] * 8,
            "avg_daily_rate": [100.0] * 8,
            "occupancy_pct": [50.0] * 8,
        }
    )
    occupancy_df = pd.DataFrame({"branch_id": [1] * 8, "date": dates, "occupancy_pct": [50.0] * 8})
    restaurant_df = pd.DataFrame({"branch_id": [1] * 8, "date": dates, "total_orders": [100] * 8})
    staff_df = pd.DataFrame(
        {
            "branch_id": [1] * 8,
            "department_name": ["Kitchen"] * 8,
            "date": dates,
            "attendance_rate_pct": [95.0] * 8,
            "scheduled_employees": [5] * 8,
            "present_employees": [5] * 8,
        }
    )

    def _fake_mart(name):
        return {
            "mart_revenue_daily": revenue_df,
            "mart_occupancy_daily": occupancy_df,
            "mart_restaurant_daily": restaurant_df,
            "mart_staff_daily": staff_df,
        }[name]

    monkeypatch.setattr(insights_service, "get_mart", _fake_mart)
    monkeypatch.setattr(insights_service, "_collect_churn_findings", lambda guest_ids=None: [])
    monkeypatch.setattr(insights_service, "_collect_review_findings", lambda: [])

    findings = insights_service.generate_findings()
    assert isinstance(findings, list)
