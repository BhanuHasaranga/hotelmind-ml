"""Generic statistical anomaly rule: flags values more than N standard
deviations from a metric's trailing mean, independent of the
category-specific threshold rules above. Useful as a catch-all for
unexpected patterns the fixed-threshold rules don't cover."""

import numpy as np
import pandas as pd

from genai.insights.rules.base import Finding

Z_SCORE_THRESHOLD = 2.5


def evaluate(df: pd.DataFrame, metric_col: str, group_col: str = "branch_id", category: str = "anomaly") -> list[Finding]:
    findings: list[Finding] = []
    if df.empty or metric_col not in df.columns:
        return findings

    for branch_id, group in df.groupby(group_col):
        values = group[metric_col].dropna().to_numpy()
        if len(values) < 10:
            continue
        mean = values[:-1].mean()
        std = values[:-1].std()
        if std == 0:
            continue
        latest = values[-1]
        z_score = (latest - mean) / std

        if abs(z_score) >= Z_SCORE_THRESHOLD:
            findings.append(
                Finding(
                    category=category,
                    title=f"Statistical anomaly in {metric_col} at branch {branch_id}",
                    metric=metric_col,
                    metric_delta=round(float(latest - mean), 2),
                    severity="high" if abs(z_score) >= 3.5 else "medium",
                    branch_id=branch_id,
                    supporting_data={"z_score": round(float(z_score), 2), "latest": float(latest), "mean": round(float(mean), 2)},
                    citation="anomaly_detector",
                )
            )
    return findings
