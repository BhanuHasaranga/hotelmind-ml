"""Occupancy rules: flags very low (underutilization risk) or very high
(overbooking/turn-away risk) occupancy, and sharp week-over-week swings.
"""

import pandas as pd

from genai.insights.rules.base import Finding

LOW_OCCUPANCY_THRESHOLD = 40.0
HIGH_OCCUPANCY_THRESHOLD = 95.0
WOW_SWING_THRESHOLD_PP = 15.0


def evaluate(occupancy_df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    if occupancy_df.empty:
        return findings

    for branch_id, group in occupancy_df.groupby("branch_id"):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        occ = float(latest["occupancy_pct"])

        if occ <= LOW_OCCUPANCY_THRESHOLD:
            findings.append(
                Finding(
                    category="occupancy",
                    title=f"Low occupancy at branch {branch_id}",
                    metric="occupancy_pct",
                    metric_delta=round(occ - LOW_OCCUPANCY_THRESHOLD, 2),
                    severity="high" if occ <= 25 else "medium",
                    branch_id=branch_id,
                    supporting_data={"occupancy_pct": occ},
                    citation="mart_occupancy_daily",
                )
            )
        elif occ >= HIGH_OCCUPANCY_THRESHOLD:
            findings.append(
                Finding(
                    category="occupancy",
                    title=f"Near-capacity occupancy at branch {branch_id}",
                    metric="occupancy_pct",
                    metric_delta=round(occ - HIGH_OCCUPANCY_THRESHOLD, 2),
                    severity="medium",
                    branch_id=branch_id,
                    supporting_data={"occupancy_pct": occ},
                    citation="mart_occupancy_daily",
                )
            )

        if len(group) >= 8:
            week_ago = float(group.iloc[-8]["occupancy_pct"])
            swing = occ - week_ago
            if abs(swing) >= WOW_SWING_THRESHOLD_PP:
                findings.append(
                    Finding(
                        category="occupancy",
                        title=f"Sharp week-over-week occupancy swing at branch {branch_id}",
                        metric="occupancy_pct_wow",
                        metric_delta=round(swing, 2),
                        severity="medium",
                        branch_id=branch_id,
                        supporting_data={"current": occ, "week_ago": week_ago},
                        citation="mart_occupancy_daily",
                    )
                )
    return findings
