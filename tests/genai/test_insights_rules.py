import pandas as pd
import pytest

from genai.insights.priority import filter_by_category, filter_by_min_severity, rank_findings, top_n
from genai.insights.rules import anomaly, churn, guest_experience, occupancy, pricing, restaurant_waste, revenue, staff
from genai.insights.rules.base import Finding
from genai.insights.scoring import score_finding, score_findings


# --- revenue ---


def test_revenue_rule_flags_drop():
    dates = pd.date_range("2026-01-01", periods=8)
    df = pd.DataFrame(
        {
            "branch_id": [1] * 8,
            "date": dates.astype(str),
            "total_revenue": [1000] * 7 + [700],
            "revenue_7day_avg": [1000] * 7 + [1000],
        }
    )
    findings = revenue.evaluate(df)
    assert any(f.title.startswith("Revenue drop") for f in findings)


def test_revenue_rule_flags_surge():
    df = pd.DataFrame(
        {
            "branch_id": [1, 1],
            "date": ["2026-01-01", "2026-01-02"],
            "total_revenue": [1000, 1300],
            "revenue_7day_avg": [1000, 1000],
        }
    )
    findings = revenue.evaluate(df)
    assert any("surge" in f.title for f in findings)


def test_revenue_rule_empty_df():
    assert revenue.evaluate(pd.DataFrame()) == []


# --- occupancy ---


def test_occupancy_rule_flags_low_occupancy():
    df = pd.DataFrame({"branch_id": [1], "date": ["2026-01-01"], "occupancy_pct": [20.0]})
    findings = occupancy.evaluate(df)
    assert any("Low occupancy" in f.title for f in findings)
    assert findings[0].severity == "high"


def test_occupancy_rule_flags_high_occupancy():
    df = pd.DataFrame({"branch_id": [1], "date": ["2026-01-01"], "occupancy_pct": [97.0]})
    findings = occupancy.evaluate(df)
    assert any("Near-capacity" in f.title for f in findings)


def test_occupancy_rule_flags_wow_swing():
    dates = pd.date_range("2026-01-01", periods=8).astype(str)
    values = [50.0] * 7 + [75.0]
    df = pd.DataFrame({"branch_id": [1] * 8, "date": dates, "occupancy_pct": values})
    findings = occupancy.evaluate(df)
    assert any("swing" in f.title for f in findings)


def test_occupancy_rule_empty():
    assert occupancy.evaluate(pd.DataFrame()) == []


# --- pricing ---


def test_pricing_rule_flags_underpricing():
    dates = pd.date_range("2026-01-01", periods=8).astype(str)
    df = pd.DataFrame(
        {
            "branch_id": [1] * 8,
            "date": dates,
            "occupancy_pct": [90.0] * 8,
            "avg_daily_rate": [100.0] * 8,
        }
    )
    findings = pricing.evaluate(df)
    assert any("under-pricing" in f.title for f in findings)


def test_pricing_rule_no_flag_when_short_history():
    df = pd.DataFrame({"branch_id": [1], "date": ["2026-01-01"], "occupancy_pct": [90.0], "avg_daily_rate": [100.0]})
    assert pricing.evaluate(df) == []


def test_pricing_rule_missing_column():
    assert pricing.evaluate(pd.DataFrame({"branch_id": [1]})) == []


# --- restaurant_waste ---


def test_restaurant_rule_flags_deviation():
    dates = pd.date_range("2026-01-01", periods=8).astype(str)
    orders = [100] * 7 + [160]
    df = pd.DataFrame({"branch_id": [1] * 8, "date": dates, "total_orders": orders})
    findings = restaurant_waste.evaluate(df)
    assert len(findings) == 1
    assert findings[0].category == "restaurant_waste"


def test_restaurant_rule_no_deviation():
    dates = pd.date_range("2026-01-01", periods=8).astype(str)
    orders = [100] * 8
    df = pd.DataFrame({"branch_id": [1] * 8, "date": dates, "total_orders": orders})
    assert restaurant_waste.evaluate(df) == []


# --- staff ---


def test_staff_rule_flags_low_attendance():
    df = pd.DataFrame(
        {
            "branch_id": [1],
            "department_name": ["Kitchen"],
            "date": ["2026-01-01"],
            "attendance_rate_pct": [60.0],
            "scheduled_employees": [5],
            "present_employees": [3],
        }
    )
    findings = staff.evaluate(df)
    categories = {f.title for f in findings}
    assert any("Low attendance" in t for t in categories)
    assert any("Understaffing" in t for t in categories)


def test_staff_rule_empty():
    assert staff.evaluate(pd.DataFrame()) == []


# --- churn ---


def test_churn_rule_flags_high_risk_high_value():
    records = [{"guest_id": "g1", "churn_probability": 0.8, "total_nights": 15}]
    findings = churn.evaluate(records)
    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_churn_rule_flags_medium_for_low_value():
    records = [{"guest_id": "g2", "churn_probability": 0.7, "total_nights": 2}]
    findings = churn.evaluate(records)
    assert findings[0].severity == "medium"


def test_churn_rule_skips_low_risk():
    records = [{"guest_id": "g3", "churn_probability": 0.1, "total_nights": 20}]
    assert churn.evaluate(records) == []


def test_churn_rule_skips_none_probability():
    records = [{"guest_id": "g4", "churn_probability": None}]
    assert churn.evaluate(records) == []


# --- guest_experience ---


def test_guest_experience_rule_flags_low_csat():
    csat_hotel = [{"hotel_id": "H1", "csat": 40.0}]
    findings = guest_experience.evaluate(csat_hotel, [], {})
    assert findings[0].severity == "high"


def test_guest_experience_rule_flags_declining_trend():
    csat_hotel = [{"hotel_id": "H1", "csat": 90.0}]
    findings = guest_experience.evaluate(csat_hotel, [], {"H1": "declining"})
    assert any("Declining" in f.title for f in findings)


def test_guest_experience_rule_flags_complaint_spike():
    complaints = [{"category": "cleanliness", "count": 60}]
    findings = guest_experience.evaluate([], complaints, {})
    assert findings[0].metric == "complaint_count"


# --- anomaly ---


def test_anomaly_rule_flags_outlier():
    baseline = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98, 101, 99, 103, 97, 100]
    values = baseline + [300]
    df = pd.DataFrame({"branch_id": [1] * len(values), "total_revenue": values})
    findings = anomaly.evaluate(df, "total_revenue")
    assert len(findings) == 1


def test_anomaly_rule_no_outlier_with_zero_std():
    df = pd.DataFrame({"branch_id": [1] * 12, "total_revenue": [100] * 12})
    assert anomaly.evaluate(df, "total_revenue") == []


def test_anomaly_rule_insufficient_data():
    df = pd.DataFrame({"branch_id": [1] * 3, "total_revenue": [100, 200, 300]})
    assert anomaly.evaluate(df, "total_revenue") == []


def test_anomaly_rule_missing_column():
    assert anomaly.evaluate(pd.DataFrame({"branch_id": [1]}), "missing_col") == []


# --- scoring & priority ---


def test_score_finding_weights_severity():
    low = Finding(category="x", title="t", metric="m", metric_delta=1.0, severity="low")
    critical = Finding(category="x", title="t", metric="m", metric_delta=1.0, severity="critical")
    assert score_finding(critical) > score_finding(low)


def test_score_finding_handles_none_delta():
    f = Finding(category="x", title="t", metric="m", metric_delta=None, severity="medium")
    assert score_finding(f) == 2.0


def test_score_findings_returns_pairs():
    findings = [Finding(category="x", title="t", metric="m", metric_delta=5.0, severity="high")]
    scored = score_findings(findings)
    assert scored[0]["finding"] is findings[0]
    assert scored[0]["score"] > 0


def test_rank_findings_orders_by_score_desc():
    low = Finding(category="a", title="low", metric="m", metric_delta=0.0, severity="low")
    high = Finding(category="b", title="high", metric="m", metric_delta=0.0, severity="high")
    ranked = rank_findings([low, high])
    assert ranked[0]["finding"] is high


def test_top_n_limits_results():
    findings = [Finding(category="x", title=f"t{i}", metric="m", metric_delta=float(i), severity="medium") for i in range(10)]
    assert len(top_n(findings, n=3)) == 3


def test_filter_by_category():
    findings = [
        Finding(category="revenue", title="a", metric="m", metric_delta=1.0, severity="low"),
        Finding(category="staff", title="b", metric="m", metric_delta=1.0, severity="low"),
    ]
    result = filter_by_category(findings, "staff")
    assert len(result) == 1
    assert result[0].category == "staff"


def test_filter_by_min_severity():
    findings = [
        Finding(category="x", title="a", metric="m", metric_delta=1.0, severity="low"),
        Finding(category="x", title="b", metric="m", metric_delta=1.0, severity="critical"),
    ]
    result = filter_by_min_severity(findings, min_severity="high")
    assert len(result) == 1
    assert result[0].severity == "critical"
