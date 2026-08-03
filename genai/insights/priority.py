"""Ranks scored findings by priority (severity score, descending), with a
stable secondary sort by category for deterministic output."""

from genai.insights.rules.base import Finding
from genai.insights.scoring import score_findings


def rank_findings(findings: list[Finding]) -> list[dict]:
    scored = score_findings(findings)
    scored.sort(key=lambda item: (-item["score"], item["finding"].category))
    return scored


def top_n(findings: list[Finding], n: int = 5) -> list[dict]:
    return rank_findings(findings)[:n]


def filter_by_category(findings: list[Finding], category: str) -> list[Finding]:
    return [f for f in findings if f.category == category]


def filter_by_min_severity(findings: list[Finding], min_severity: str = "medium") -> list[Finding]:
    order = ["low", "medium", "high", "critical"]
    threshold_idx = order.index(min_severity)
    return [f for f in findings if order.index(f.severity) >= threshold_idx]
