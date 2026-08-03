"""Guest experience rules: derived from Module 1 (review analysis) CSAT and
complaint outputs."""

from genai.insights.rules.base import Finding

LOW_CSAT_THRESHOLD = 60.0
COMPLAINT_SPIKE_COUNT = 50


def evaluate(csat_hotel: list[dict], complaint_counts: list[dict], trend_by_hotel: dict) -> list[Finding]:
    findings: list[Finding] = []

    for row in csat_hotel:
        csat = row.get("csat")
        hotel_id = row.get("hotel_id")
        if csat is not None and csat < LOW_CSAT_THRESHOLD:
            findings.append(
                Finding(
                    category="guest_experience",
                    title=f"Low guest satisfaction at {hotel_id}",
                    metric="csat",
                    metric_delta=round(csat - LOW_CSAT_THRESHOLD, 2),
                    severity="high" if csat < 45 else "medium",
                    branch_id=hotel_id,
                    supporting_data={"csat": csat},
                    citation="guest_review_analysis",
                )
            )
        trend = trend_by_hotel.get(hotel_id)
        if trend == "declining":
            findings.append(
                Finding(
                    category="guest_experience",
                    title=f"Declining satisfaction trend at {hotel_id}",
                    metric="csat_trend",
                    metric_delta=None,
                    severity="medium",
                    branch_id=hotel_id,
                    supporting_data={"trend": trend},
                    citation="guest_review_analysis",
                )
            )

    for row in complaint_counts:
        count = row.get("count", 0)
        category = row.get("category")
        if count >= COMPLAINT_SPIKE_COUNT:
            findings.append(
                Finding(
                    category="guest_experience",
                    title=f"High volume of '{category}' complaints",
                    metric="complaint_count",
                    metric_delta=float(count),
                    severity="high" if count >= COMPLAINT_SPIKE_COUNT * 2 else "medium",
                    supporting_data={"category": category, "count": count},
                    citation="guest_review_analysis",
                )
            )
    return findings
