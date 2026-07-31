"""Part 1 orchestration: builds a standalone, inspectable feature dataset
per domain, independent of any training run, and saves it to
data/features/{domain}_features.parquet.

Reuses each domain's existing src/features/*.py functions -- never
duplicates logic already in the *_pipeline.py classes. Kept separate from
those classes because their engineer_features() is scoped to that
pipeline's own train/test flow (called inside run()), while this module's
purpose is to produce a reviewable dataset on its own.

CLI: python -m src.pipelines.feature_engineering
"""

import time

import pandas as pd

from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.features.churn_features import add_rfm_features, label_churn
from src.features.occupancy_aggregation import load_daily_occupancy
from src.features.preprocessing import encode_categoricals, handle_missing_values
from src.features.pricing_features import add_demand_index, add_room_types
from src.features.restaurant_features import derive_meal_quantities
from src.features.time_series_features import add_lag_features, add_rolling_features
from src.pipelines.ml_reports import (
    write_correlation_report,
    write_feature_dictionary,
    write_feature_statistics,
)
from src.pipelines.synthetic_data import ensure_restaurant_seed, ensure_staffing_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_occupancy_features() -> pd.DataFrame:
    df = load_daily_occupancy()
    df = add_calendar_features(df, "occupancy_date")
    df = add_lag_features(df, "occupancy_pct", lags=[1, 7, 30], group_col="branch_id")
    df = add_rolling_features(df, "occupancy_pct", windows=[7, 30], group_col="branch_id")
    df["occupancy_trend"] = df["occupancy_pct"] - df["occupancy_7day_avg"]
    df["revenue_trend"] = df["total_revenue"] - df["revenue_7day_avg"]
    df = handle_missing_values(
        df, strategy="median", columns=[c for c in df.columns if "lag" in c or "rolling" in c]
    )
    return df


def build_pricing_features() -> pd.DataFrame:
    df = load_daily_occupancy()
    df = df.rename(columns={"occupancy_date": "date"})
    df = add_calendar_features(df, "date")
    df = add_demand_index(df)
    df = add_room_types(df)
    df, _ = encode_categoricals(df, ["season", "room_type_name"])
    return df


def build_restaurant_features() -> pd.DataFrame:
    daily_occupancy = load_daily_occupancy()
    seed_path = ensure_restaurant_seed(daily_occupancy)
    df = pd.read_csv(seed_path, parse_dates=["date"])
    df = add_calendar_features(df, "date")
    df = derive_meal_quantities(df)
    df = add_lag_features(df, "total_orders", lags=[1, 7], group_col="branch_id")
    df = add_rolling_features(df, "total_orders", windows=[7], group_col="branch_id")
    df = handle_missing_values(
        df, strategy="median", columns=[c for c in df.columns if "lag" in c or "rolling" in c]
    )
    return df


def build_staff_features() -> pd.DataFrame:
    daily_occupancy = load_daily_occupancy()
    seed_path = ensure_staffing_seed(daily_occupancy)
    df = pd.read_csv(seed_path, parse_dates=["date"])
    df = df.sort_values(["department_id", "date"]).reset_index(drop=True)
    df = add_calendar_features(df, "date")
    df = add_lag_features(df, "present_employees", lags=[1, 7, 30], group_col="department_id")
    df = add_rolling_features(df, "present_employees", windows=[7, 30], group_col="department_id")
    df, _ = encode_categoricals(df, ["department_name"])
    df = handle_missing_values(
        df, strategy="median", columns=[c for c in df.columns if "lag" in c or "rolling" in c]
    )
    return df


def build_churn_features() -> pd.DataFrame:
    dim_guest = pd.read_parquet(settings.data_warehouse_dir_path / "dim_guest.parquet")
    fact_booking = pd.read_parquet(settings.data_warehouse_dir_path / "fact_booking.parquet")

    total_nights = fact_booking.groupby("guest_key")["nights"].sum().rename("total_nights")
    df = dim_guest.merge(total_nights, left_on="guest_key", right_index=True, how="left")
    df["total_nights"] = df["total_nights"].fillna(0)

    cancellations = (
        fact_booking.assign(is_cancel=fact_booking["reservation_status"] == "Canceled")
        .groupby("guest_key")["is_cancel"].mean()
        .rename("cancellation_ratio")
    )
    df = df.merge(cancellations, left_on="guest_key", right_index=True, how="left")
    df["cancellation_ratio"] = df["cancellation_ratio"].fillna(0.0)
    df["repeat_guest"] = (df["lifetime_bookings"] > 1).astype(int)

    # See src/pipelines/churn_pipeline.py for why snapshot_date is anchored
    # to the dataset's own date range rather than the real system clock.
    snapshot_date = (df["last_stay_date"].max() + pd.Timedelta(days=1)).date()
    df = label_churn(df, snapshot_date=snapshot_date)
    df = add_rfm_features(df)
    return df


DOMAIN_BUILDERS = {
    "occupancy": build_occupancy_features,
    "pricing": build_pricing_features,
    "restaurant": build_restaurant_features,
    "staff": build_staff_features,
    "churn": build_churn_features,
}


def run() -> dict[str, pd.DataFrame]:
    start = time.perf_counter()
    feature_dfs: dict[str, pd.DataFrame] = {}

    for domain, builder in DOMAIN_BUILDERS.items():
        df = builder()
        path = settings.data_features_dir_path / f"{domain}_features.parquet"
        df.to_parquet(path, index=False)
        feature_dfs[domain] = df
        logger.info("Built %s features: %d rows, %d cols -> %s", domain, len(df), df.shape[1], path)

    write_feature_dictionary(feature_dfs)
    write_feature_statistics(feature_dfs)
    write_correlation_report(feature_dfs)

    elapsed = time.perf_counter() - start
    logger.info("feature_engineering complete in %.2fs", elapsed)
    return feature_dfs


if __name__ == "__main__":
    run()
