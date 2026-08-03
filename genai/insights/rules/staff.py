"""Staffing rules: flags departments with attendance/coverage gaps."""

import pandas as pd

from genai.insights.rules.base import Finding

LOW_ATTENDANCE_THRESHOLD_PCT = 80.0
UNDERSTAFFED_GAP_THRESHOLD = 2


def evaluate(staff_df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    if staff_df.empty:
        return findings

    group_cols = ["branch_id", "department_name"] if "department_name" in staff_df.columns else ["branch_id"]
    for keys, group in staff_df.groupby(group_cols):
        branch_id = keys[0] if isinstance(keys, tuple) else keys
        department = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else None
        group = group.sort_values("date")
        latest = group.iloc[-1]

        attendance = float(latest.get("attendance_rate_pct", 100) or 100)
        gap = int(latest.get("scheduled_employees", 0) - latest.get("present_employees", 0))

        if attendance < LOW_ATTENDANCE_THRESHOLD_PCT:
            findings.append(
                Finding(
                    category="staff",
                    title=f"Low attendance at branch {branch_id}" + (f" ({department})" if department else ""),
                    metric="attendance_rate_pct",
                    metric_delta=round(attendance - LOW_ATTENDANCE_THRESHOLD_PCT, 2),
                    severity="high" if attendance < 65 else "medium",
                    branch_id=branch_id,
                    supporting_data={"attendance_rate_pct": attendance, "department": department},
                    citation="mart_staff_daily",
                )
            )
        if gap >= UNDERSTAFFED_GAP_THRESHOLD:
            findings.append(
                Finding(
                    category="staff",
                    title=f"Understaffing gap at branch {branch_id}" + (f" ({department})" if department else ""),
                    metric="staffing_gap",
                    metric_delta=float(gap),
                    severity="medium" if gap < 4 else "high",
                    branch_id=branch_id,
                    supporting_data={"gap": gap, "department": department},
                    citation="mart_staff_daily",
                )
            )
    return findings
