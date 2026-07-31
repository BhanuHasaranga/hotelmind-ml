import pandas as pd

from src.pipelines.occupancy_pipeline import OccupancyPipeline
from src.pipelines.churn_pipeline import ChurnPipeline


def _sample_occupancy_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "branch_id": 1,
            "branch_name": "Test Branch",
            "occupancy_date": dates,
            "year": dates.year,
            "month": dates.month,
            "quarter": dates.quarter,
            "week_of_year": dates.isocalendar().week.to_numpy(),
            "day_of_week": dates.dayofweek,
            "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
            "total_rooms": 100,
            "occupied_rooms": 60,
            "available_rooms": 40,
            "occupancy_pct": [50 + (i % 10) for i in range(60)],
            "occupancy_7day_avg": 55.0,
            "occupancy_30day_avg": 55.0,
            "occupancy_pct_lag_7d": 50.0,
            "occupancy_pct_lag_365d": 50.0,
            "occupancy_mtd_avg": 55.0,
        }
    )


def test_occupancy_pipeline_clean_and_engineer():
    pipeline = OccupancyPipeline(branch_id=1, start_date="2024-01-01", end_date="2024-03-01")
    df = _sample_occupancy_df()

    cleaned = pipeline.clean(df)
    assert cleaned["occupancy_pct"].isna().sum() == 0

    featured = pipeline.engineer_features(cleaned)
    for col in ["month", "is_weekend", "season", "occupancy_pct_lag_1", "occupancy_pct_rolling_mean_7"]:
        assert col in featured.columns


def _sample_guest_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "guest_key": [1, 2, 3],
            "guest_id": ["g1", "g2", "g3"],
            "full_name": ["A", "B", "C"],
            "nationality": ["US", "US", "US"],
            "lifetime_bookings": [3, 1, 0],
            "lifetime_spend": [900.0, 100.0, 0.0],
            "first_stay_date": pd.to_datetime(["2023-01-01", "2024-01-01", pd.NaT]),
            "last_stay_date": pd.to_datetime(["2024-01-01", "2025-06-01", pd.NaT]),
            "completed_bookings": [3, 1, 0],
            "total_nights": [9, 2, 0],
            "avg_spend_per_stay": [300.0, 100.0, 0.0],
        }
    )


def test_churn_pipeline_label_and_features():
    pipeline = ChurnPipeline()
    df = _sample_guest_df()

    cleaned = pipeline.clean(df)
    # guest 3 has no last_stay_date -> dropped
    assert len(cleaned) == 2

    featured = pipeline.engineer_features(cleaned)
    assert "churn" in featured.columns
    assert "recency_days" in featured.columns
    assert featured["churn"].isin([0, 1]).all()
