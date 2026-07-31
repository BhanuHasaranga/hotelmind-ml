import datetime as dt

import pandas as pd

from src.config.settings import settings


def label_churn(
    df: pd.DataFrame,
    snapshot_date: dt.date | None = None,
    window_days: int | None = None,
) -> pd.DataFrame:
    """Derive a binary churn label from dim_guest recency data.

    ASSUMPTION (locked in with user): churn = 1 if
    (snapshot_date - last_stay_date).days > window_days AND
    lifetime_bookings >= 1. window_days defaults to settings.CHURN_WINDOW_DAYS
    (180). snapshot_date defaults to today. Guests with lifetime_bookings == 0
    (never stayed) are excluded — churn is only meaningful for guests who have
    stayed at least once. See README "Assumptions & Data Gaps".
    """
    snapshot_date = snapshot_date or dt.date.today()
    window_days = window_days or settings.CHURN_WINDOW_DAYS

    out = df.copy()
    out["last_stay_date"] = pd.to_datetime(out["last_stay_date"])
    out = out[out["lifetime_bookings"] >= 1].reset_index(drop=True)

    recency_days = (pd.Timestamp(snapshot_date) - out["last_stay_date"]).dt.days
    out["recency_days"] = recency_days
    out["churn"] = (recency_days > window_days).astype(int)
    return out


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["frequency"] = out["lifetime_bookings"]
    out["monetary"] = out["lifetime_spend"]
    out["avg_spend_per_stay"] = (out["lifetime_spend"] / out["lifetime_bookings"]).replace(
        [float("inf"), -float("inf")], 0.0
    ).fillna(0.0)
    return out
