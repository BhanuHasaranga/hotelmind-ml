import pandas as pd

from src.config.constants import MONTH_TO_SEASON
from src.config.settings import settings

_EVENTS_CALENDAR_PATH = settings.data_raw_dir_path / "events_holiday_calendar.csv"


def _load_events_calendar() -> pd.DataFrame:
    df = pd.read_csv(_EVENTS_CALENDAR_PATH, parse_dates=["date"])
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add month/quarter/day_of_week/weekend/season/holiday/event columns.

    is_holiday / is_event / event_name come from the synthetic
    data/raw/events_holiday_calendar.csv seed (see README "Assumptions & Data Gaps")
    since the Phase 3 warehouse has no holiday/events dimension.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    out["month"] = out[date_col].dt.month
    out["quarter"] = out[date_col].dt.quarter
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["season"] = out["month"].map(MONTH_TO_SEASON)

    events = _load_events_calendar()
    out = out.merge(events, left_on=date_col, right_on="date", how="left")
    out["is_public_holiday"] = out["is_public_holiday"].fillna(0).astype(int)
    out["is_local_event"] = out["is_local_event"].fillna(0).astype(int)
    out["event_name"] = out["event_name"].fillna("")
    out["is_holiday"] = out["is_public_holiday"]
    out["is_event"] = out["is_local_event"]
    if "date" in out.columns and "date" != date_col:
        out = out.drop(columns=["date"])

    return out
