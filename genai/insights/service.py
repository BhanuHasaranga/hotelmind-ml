"""Consumes ML prediction outputs, review analysis outputs, and warehouse
marts to produce structured JSON findings with LLM-generated recommendation
text — gracefully degrading to a rule-based template when no LLM
provider/key is available.
"""

from __future__ import annotations

import json

import pandas as pd

from genai.data_access.warehouse_source import get_mart
from genai.insights.priority import rank_findings
from genai.insights.rules import anomaly, churn, guest_experience, occupancy, pricing, restaurant_waste, revenue, staff
from genai.insights.rules.base import Finding
from genai.prompts.loader import load_prompt
from genai.reviews.service import ReviewsNotAnalyzedError, get_complaints, get_summary, get_trends
from src.config.settings import settings
from src.prediction.predict_churn import predict_churn
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _rule_based_recommendation(finding: Finding) -> str:
    """Deterministic template used when no LLM provider is
    available/configured. Never hard-fails."""
    delta_text = f" (delta: {finding.metric_delta})" if finding.metric_delta is not None else ""
    return (
        f"[{finding.severity.upper()}] {finding.title}{delta_text}. "
        f"Review {finding.metric} for {finding.citation or 'the relevant source'} "
        f"and take corrective action if this persists beyond the next reporting cycle."
    )


def _llm_recommendation(finding: Finding, llm_provider) -> str:
    if llm_provider is None or not llm_provider.is_available():
        return _rule_based_recommendation(finding)
    try:
        system_prompt = load_prompt("executive_assistant").text
        prompt = (
            f"Category: {finding.category}\nTitle: {finding.title}\n"
            f"Metric: {finding.metric}\nDelta: {finding.metric_delta}\n"
            f"Severity: {finding.severity}\nSupporting data: {finding.supporting_data}\n\n"
            "Write a 1-2 sentence, business-actionable recommendation for a hotel "
            "operations manager based on this finding."
        )
        result = llm_provider.generate(prompt, system=system_prompt)
        return result.text.strip()
    except Exception as exc:
        logger.warning("LLM recommendation generation failed, using rule-based fallback: %s", exc)
        return _rule_based_recommendation(finding)


def _collect_churn_findings(guest_ids: list[str] | None = None) -> list[Finding]:
    if not guest_ids:
        try:
            dim_guest = pd.read_parquet(settings.data_warehouse_dir_path / "dim_guest.parquet")
            guest_ids = dim_guest["guest_id"].astype(str).head(20).tolist()
        except Exception as exc:
            logger.warning("Could not sample guests for churn insights: %s", exc)
            return []

    records = []
    for guest_id in guest_ids:
        try:
            result = predict_churn(guest_id)
            records.append(result)
        except Exception as exc:
            logger.debug("Skipping churn insight for guest %s: %s", guest_id, exc)
    return churn.evaluate(records)


def _collect_review_findings() -> list[Finding]:
    try:
        summary = get_summary()
        complaints = get_complaints()
        trends = get_trends(grain="weekly")
    except ReviewsNotAnalyzedError:
        logger.info("Review analysis not yet run; skipping guest_experience findings")
        return []
    return guest_experience.evaluate(
        summary.get("csat_by_hotel", []), complaints.get("complaints", []), trends.get("trend_by_hotel", {})
    )


def generate_findings(guest_ids_for_churn: list[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []

    revenue_df = get_mart("mart_revenue_daily")
    occupancy_df = get_mart("mart_occupancy_daily")
    restaurant_df = get_mart("mart_restaurant_daily")
    staff_df = get_mart("mart_staff_daily")

    findings.extend(revenue.evaluate(revenue_df))
    findings.extend(occupancy.evaluate(occupancy_df))
    findings.extend(pricing.evaluate(revenue_df))
    findings.extend(restaurant_waste.evaluate(restaurant_df))
    findings.extend(staff.evaluate(staff_df))
    findings.extend(anomaly.evaluate(revenue_df, "total_revenue", category="anomaly"))
    findings.extend(_collect_churn_findings(guest_ids_for_churn))
    findings.extend(_collect_review_findings())

    return findings


def _finding_to_dict(finding: Finding, score: float, recommendation: str) -> dict:
    return {
        "category": finding.category,
        "title": finding.title,
        "metric": finding.metric,
        "metric_delta": finding.metric_delta,
        "severity": finding.severity,
        "branch_id": finding.branch_id,
        "supporting_data": finding.supporting_data,
        "citation": finding.citation,
        "priority_score": score,
        "recommendation": recommendation,
    }


def get_insights(category: str | None = None, min_severity: str | None = None, llm_provider=None) -> list[dict]:
    findings = generate_findings()
    if category:
        findings = [f for f in findings if f.category == category]
    if min_severity:
        from genai.insights.priority import filter_by_min_severity

        findings = filter_by_min_severity(findings, min_severity)

    ranked = rank_findings(findings)
    return [
        _finding_to_dict(item["finding"], item["score"], _llm_recommendation(item["finding"], llm_provider))
        for item in ranked
    ]


def get_executive_summary(llm_provider=None, top_n_count: int = 5) -> dict:
    findings = generate_findings()
    ranked = rank_findings(findings)[:top_n_count]
    items = [
        _finding_to_dict(item["finding"], item["score"], _llm_recommendation(item["finding"], llm_provider))
        for item in ranked
    ]

    if llm_provider is not None and llm_provider.is_available() and items:
        try:
            system_prompt = load_prompt("executive_assistant").text
            prompt = "Top findings:\n" + json.dumps(items, default=str) + "\n\nWrite a short executive briefing."
            result = llm_provider.generate(prompt, system=system_prompt)
            narrative = result.text
        except Exception as exc:
            logger.warning("Executive narrative generation failed, using template: %s", exc)
            narrative = _template_executive_summary(items)
    else:
        narrative = _template_executive_summary(items)

    return {"narrative": narrative, "top_findings": items}


def _template_executive_summary(items: list[dict]) -> str:
    if not items:
        return "No significant findings at this time."
    lines = [f"- [{i['severity'].upper()}] {i['title']}" for i in items]
    return "Top findings requiring attention:\n" + "\n".join(lines)


def get_recommendations(llm_provider=None) -> list[dict]:
    findings = generate_findings()
    ranked = rank_findings(findings)
    return [
        {
            "category": item["finding"].category,
            "title": item["finding"].title,
            "recommendation": _llm_recommendation(item["finding"], llm_provider),
            "priority_score": item["score"],
        }
        for item in ranked
    ]


def get_anomalies(llm_provider=None) -> list[dict]:
    revenue_df = get_mart("mart_revenue_daily")
    occupancy_df = get_mart("mart_occupancy_daily")
    findings = anomaly.evaluate(revenue_df, "total_revenue", category="anomaly")
    findings += anomaly.evaluate(occupancy_df, "occupancy_pct", category="anomaly")
    ranked = rank_findings(findings)
    return [
        _finding_to_dict(item["finding"], item["score"], _llm_recommendation(item["finding"], llm_provider))
        for item in ranked
    ]
