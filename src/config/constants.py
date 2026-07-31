"""Shared constants used across ML modules.

Season and risk-level bucketing are business-defined, reasonable defaults
(not derived from data) — documented here as the single source of truth.
"""

# Meteorological-style season mapping used for calendar feature engineering.
MONTH_TO_SEASON: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

STAFF_DEPARTMENTS: list[str] = ["Reception", "Kitchen", "Housekeeping"]

MEAL_PERIODS: list[str] = ["breakfast", "lunch", "dinner"]

# Churn risk-level thresholds applied to predicted churn probability.
CHURN_RISK_THRESHOLDS: dict[str, float] = {
    "low_max": 0.3,
    "medium_max": 0.6,
}


def churn_probability_to_risk_level(probability: float) -> str:
    if probability < CHURN_RISK_THRESHOLDS["low_max"]:
        return "Low"
    if probability < CHURN_RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"
