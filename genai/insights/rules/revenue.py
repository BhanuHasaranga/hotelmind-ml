"""Revenue rules: flags significant drops/rises in daily revenue vs. its
trailing 7-day average, per branch.
"""

import pandas as pd

from genai.insights.rules.base import Finding

DROP_THRESHOLD_PCT = -10.0
RISE_THRESHOLD_PCT = 15.0


def evaluate(revenue_df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    if revenue_df.empty:
        return findings

    for branch_id, group in revenue_df.groupby("branch_id"):
        group = group.sort_values("date")
        if len(group) < 2:
            continue
        latest = group.iloc[-1]
        avg_7d = group["revenue_7day_avg"].iloc[-1] if "revenue_7day_avg" in group else group["total_revenue"].mean()
        if avg_7d == 0:
            continue
        pct_delta = (latest["total_revenue"] - avg_7d) / avg_7d * 100

        if pct_delta <= DROP_THRESHOLD_PCT:
            findings.append(
                Finding(
                    category="revenue",
                    title=f"Revenue drop at branch {branch_id}",
                    metric="total_revenue",
                    metric_delta=round(pct_delta, 2),
                    severity="high" if pct_delta <= -20 else "medium",
                    branch_id=branch_id,
                    supporting_data={"latest_revenue": float(latest["total_revenue"]), "avg_7d": float(avg_7d)},
                    citation="mart_revenue_daily",
                )
            )
        elif pct_delta >= RISE_THRESHOLD_PCT:
            findings.append(
                Finding(
                    category="revenue",
                    title=f"Revenue surge at branch {branch_id}",
                    metric="total_revenue",
                    metric_delta=round(pct_delta, 2),
                    severity="low",
                    branch_id=branch_id,
                    supporting_data={"latest_revenue": float(latest["total_revenue"]), "avg_7d": float(avg_7d)},
                    citation="mart_revenue_daily",
                )
            )
    return findings
