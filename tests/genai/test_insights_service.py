"""Tests for genai/insights/service.py -- findings generation, LLM
recommendation (with fallback), executive summary, and the /insights/*
service functions. All ML/warehouse/review dependencies are stubbed so
these run fully offline and deterministically."""

from __future__ import annotations

import pandas as pd
import pytest

from genai.insights import service as insights_service
from genai.insights.rules.base import Finding
from genai.insights.service import _collect_churn_findings as _real_collect_churn_findings
from genai.reviews.service import ReviewsNotAnalyzedError


def _fake_mart(name):
    dates = pd.date_range("2026-01-01", periods=8).astype(str)
    if name == "mart_revenue_daily":
        return pd.DataFrame(
            {
                "branch_id": [1] * 8,
                "date": dates,
                "total_revenue": [1000] * 7 + [600],
                "revenue_7day_avg": [1000] * 8,
                "avg_daily_rate": [100.0] * 8,
                "occupancy_pct": [90.0] * 8,
            }
        )
    if name == "mart_occupancy_daily":
        return pd.DataFrame({"branch_id": [1] * 8, "date": dates, "occupancy_pct": [90.0] * 8})
    if name == "mart_restaurant_daily":
        return pd.DataFrame({"branch_id": [1] * 8, "date": dates, "total_orders": [100] * 8})
    if name == "mart_staff_daily":
        return pd.DataFrame(
            {
                "branch_id": [1] * 8,
                "department_name": ["Kitchen"] * 8,
                "date": dates,
                "attendance_rate_pct": [95.0] * 8,
                "scheduled_employees": [5] * 8,
                "present_employees": [5] * 8,
            }
        )
    raise ValueError(name)


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch):
    monkeypatch.setattr(insights_service, "get_mart", _fake_mart)
    monkeypatch.setattr(insights_service, "predict_churn", lambda guest_id: {"churn_probability": 0.1, "guest_id": guest_id})

    def _fake_collect_review_findings():
        return []

    monkeypatch.setattr(insights_service, "_collect_review_findings", _fake_collect_review_findings)
    monkeypatch.setattr(insights_service, "_collect_churn_findings", lambda guest_ids=None: [])


def test_generate_findings_includes_revenue_drop():
    findings = insights_service.generate_findings()
    assert any(f.category == "revenue" for f in findings)


def test_get_insights_without_llm_uses_rule_based_recommendation():
    results = insights_service.get_insights(llm_provider=None)
    assert len(results) > 0
    assert all("recommendation" in r for r in results)
    assert results[0]["recommendation"]  # non-empty


def test_get_insights_filters_by_category():
    results = insights_service.get_insights(category="revenue", llm_provider=None)
    assert all(r["category"] == "revenue" for r in results)


def test_get_insights_filters_by_min_severity():
    results = insights_service.get_insights(min_severity="high", llm_provider=None)
    for r in results:
        assert r["severity"] in {"high", "critical"}


def test_get_executive_summary_without_llm_uses_template():
    result = insights_service.get_executive_summary(llm_provider=None)
    assert "narrative" in result
    assert "top_findings" in result


def test_get_executive_summary_with_no_findings_returns_no_significant():
    import genai.insights.service as svc

    original = svc.generate_findings
    svc.generate_findings = lambda *a, **k: []
    try:
        result = svc.get_executive_summary(llm_provider=None)
        assert result["narrative"] == "No significant findings at this time."
    finally:
        svc.generate_findings = original


def test_get_recommendations_without_llm():
    results = insights_service.get_recommendations(llm_provider=None)
    assert all("recommendation" in r for r in results)


def test_get_anomalies_without_llm():
    results = insights_service.get_anomalies(llm_provider=None)
    assert isinstance(results, list)


class _FakeLLM:
    def is_available(self):
        return True

    def generate(self, prompt, system=None, **kwargs):
        from genai.llm.base import LLMResult

        return LLMResult(text="LLM-generated recommendation.", provider="fake", model="fake-model")


def test_get_insights_with_available_llm_uses_llm_text():
    results = insights_service.get_insights(llm_provider=_FakeLLM())
    assert any(r["recommendation"] == "LLM-generated recommendation." for r in results)


class _BrokenLLM:
    def is_available(self):
        return True

    def generate(self, prompt, system=None, **kwargs):
        raise RuntimeError("LLM unreachable")


def test_get_insights_llm_failure_falls_back_to_rule_based():
    results = insights_service.get_insights(llm_provider=_BrokenLLM())
    assert all("[" in r["recommendation"] for r in results)  # rule-based template starts with [SEVERITY]


def test_collect_review_findings_handles_not_analyzed(monkeypatch):
    """Exercises the real `_collect_review_findings` function's
    ReviewsNotAnalyzedError handling directly (importlib.reload avoids the
    autouse fixture's module-level stub of this same function)."""
    import importlib

    import genai.insights.service as svc

    fresh_svc = importlib.reload(svc)

    def _raise_summary():
        raise ReviewsNotAnalyzedError("not yet")

    monkeypatch.setattr(fresh_svc, "get_summary", _raise_summary)
    result = fresh_svc._collect_review_findings()
    assert result == []

    # Restore module state for any tests running after this one in-process.
    importlib.reload(svc)


def test_collect_churn_findings_with_explicit_guest_ids(monkeypatch):
    """Exercises the real `_collect_churn_findings` (not the autouse stub)
    with explicit guest_ids, so the dim_guest sampling branch is skipped."""
    import genai.insights.service as svc

    monkeypatch.setattr(svc, "predict_churn", lambda guest_id: {"guest_id": guest_id, "churn_probability": 0.9, "total_nights": 20})
    real_fn = _real_collect_churn_findings
    findings = real_fn(guest_ids=["g1", "g2"])
    assert len(findings) == 2
    assert all(f.category == "churn" for f in findings)


def test_collect_churn_findings_skips_prediction_errors(monkeypatch):
    import genai.insights.service as svc

    def _raise(guest_id):
        raise ValueError("guest not found")

    monkeypatch.setattr(svc, "predict_churn", _raise)
    real_fn = _real_collect_churn_findings
    findings = real_fn(guest_ids=["ghost"])
    assert findings == []


def test_collect_churn_findings_samples_guests_when_none_given(monkeypatch):
    import pandas as pd

    import genai.insights.service as svc

    fake_dim_guest = pd.DataFrame({"guest_id": ["g1", "g2", "g3"]})
    monkeypatch.setattr(pd, "read_parquet", lambda path: fake_dim_guest)
    monkeypatch.setattr(svc, "predict_churn", lambda guest_id: {"guest_id": guest_id, "churn_probability": 0.05, "total_nights": 1})
    real_fn = _real_collect_churn_findings
    findings = real_fn(guest_ids=None)
    assert findings == []  # low churn probability -> no findings, but no error


def test_collect_churn_findings_handles_sampling_failure(monkeypatch):
    import genai.insights.service as svc

    def _raise(*a, **k):
        raise FileNotFoundError("dim_guest.parquet missing")

    import pandas as pd

    monkeypatch.setattr(pd, "read_parquet", _raise)
    real_fn = _real_collect_churn_findings
    assert real_fn(guest_ids=None) == []


def test_get_executive_summary_with_llm_uses_llm_narrative():
    result = insights_service.get_executive_summary(llm_provider=_FakeLLM())
    assert result["narrative"] == "LLM-generated recommendation."


def test_get_executive_summary_llm_failure_falls_back_to_template():
    result = insights_service.get_executive_summary(llm_provider=_BrokenLLM())
    assert "Top findings requiring attention" in result["narrative"]
