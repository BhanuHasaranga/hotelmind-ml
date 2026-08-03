"""Churn rules: flags high-risk guests emerging from the churn model,
particularly high-value guests."""

from genai.insights.rules.base import Finding

HIGH_RISK_THRESHOLD = 0.6
HIGH_VALUE_NIGHTS_THRESHOLD = 10


def evaluate(churn_records: list[dict]) -> list[Finding]:
    """`churn_records`: list of dicts with keys guest_id, churn_probability,
    risk_level, total_nights (optional)."""
    findings: list[Finding] = []
    for record in churn_records:
        probability = record.get("churn_probability")
        if probability is None or probability < HIGH_RISK_THRESHOLD:
            continue
        total_nights = record.get("total_nights", 0)
        is_high_value = total_nights >= HIGH_VALUE_NIGHTS_THRESHOLD
        findings.append(
            Finding(
                category="churn",
                title=f"High churn risk guest {record.get('guest_id')}"
                + (" (high-value)" if is_high_value else ""),
                metric="churn_probability",
                metric_delta=round(probability - HIGH_RISK_THRESHOLD, 4),
                severity="critical" if is_high_value else "medium",
                supporting_data={
                    "guest_id": record.get("guest_id"),
                    "churn_probability": probability,
                    "total_nights": total_nights,
                },
                citation="predict_churn",
            )
        )
    return findings
