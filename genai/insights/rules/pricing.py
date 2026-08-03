"""Pricing rules: flags large gaps between current ADR and RevPAR-implied
optimal pricing headroom (high occupancy but flat/low ADR growth suggests
under-pricing).
"""

import pandas as pd

from genai.insights.rules.base import Finding

UNDERPRICING_OCCUPANCY_THRESHOLD = 85.0
UNDERPRICING_ADR_GROWTH_THRESHOLD_PCT = 2.0


def evaluate(revenue_df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    if revenue_df.empty or "avg_daily_rate" not in revenue_df.columns:
        return findings

    for branch_id, group in revenue_df.groupby("branch_id"):
        group = group.sort_values("date")
        if len(group) < 8:
            continue
        latest = group.iloc[-1]
        week_ago = group.iloc[-8]
        occ = float(latest.get("occupancy_pct", 0) or 0)
        adr_growth_pct = (
            (latest["avg_daily_rate"] - week_ago["avg_daily_rate"]) / week_ago["avg_daily_rate"] * 100
            if week_ago["avg_daily_rate"]
            else 0
        )

        if occ >= UNDERPRICING_OCCUPANCY_THRESHOLD and adr_growth_pct < UNDERPRICING_ADR_GROWTH_THRESHOLD_PCT:
            findings.append(
                Finding(
                    category="pricing",
                    title=f"Potential under-pricing at branch {branch_id}",
                    metric="avg_daily_rate_growth_pct",
                    metric_delta=round(adr_growth_pct, 2),
                    severity="medium",
                    branch_id=branch_id,
                    supporting_data={"occupancy_pct": occ, "adr_growth_pct": round(adr_growth_pct, 2)},
                    citation="mart_revenue_daily",
                )
            )
    return findings
