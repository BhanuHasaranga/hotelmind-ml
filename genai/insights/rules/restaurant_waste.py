"""Restaurant waste/demand rules: flags meal periods where actual order
volume deviates sharply from the model's forecast, a proxy for over/under
prep risk."""

import pandas as pd

from genai.insights.rules.base import Finding

DEVIATION_THRESHOLD_PCT = 25.0


def evaluate(restaurant_df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    if restaurant_df.empty:
        return findings

    for branch_id, group in restaurant_df.groupby("branch_id"):
        group = group.sort_values("date")
        if len(group) < 8:
            continue
        latest = group.iloc[-1]
        avg_7d = group["total_orders"].iloc[-8:-1].mean() if "total_orders" in group else None
        if not avg_7d:
            continue
        pct_delta = (latest["total_orders"] - avg_7d) / avg_7d * 100

        if abs(pct_delta) >= DEVIATION_THRESHOLD_PCT:
            findings.append(
                Finding(
                    category="restaurant_waste",
                    title=f"Restaurant demand deviation at branch {branch_id}",
                    metric="total_orders",
                    metric_delta=round(pct_delta, 2),
                    severity="medium" if abs(pct_delta) < 40 else "high",
                    branch_id=branch_id,
                    supporting_data={"latest_orders": float(latest["total_orders"]), "avg_7d": float(avg_7d)},
                    citation="mart_restaurant_daily",
                )
            )
    return findings
