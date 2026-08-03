"""Severity scoring: converts a Finding's qualitative severity + metric
delta magnitude into a numeric score for ranking."""

from genai.insights.rules.base import Finding

_SEVERITY_WEIGHTS = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def score_finding(finding: Finding) -> float:
    base = _SEVERITY_WEIGHTS.get(finding.severity, 1)
    magnitude = abs(finding.metric_delta) if finding.metric_delta is not None else 0.0
    # Normalize magnitude contribution so it nudges but never overwhelms
    # the base severity weight (log-scaled to dampen large deltas like
    # complaint counts vs. small deltas like percentage points).
    import math

    magnitude_score = math.log1p(magnitude) / 10
    return round(base + magnitude_score, 4)


def score_findings(findings: list[Finding]) -> list[dict]:
    return [
        {
            "finding": finding,
            "score": score_finding(finding),
        }
        for finding in findings
    ]
