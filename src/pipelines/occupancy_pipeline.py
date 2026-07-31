from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.features.preprocessing import handle_missing_values
from src.features.time_series_features import add_lag_features, add_rolling_features
from src.models.base import BaseMLModel
from src.models.occupancy.prophet_model import OccupancyProphetModel
from src.models.occupancy.xgboost_model import OccupancyXGBoostModel
from src.pipelines.base_pipeline import BasePipeline, default_time_series_split

DATE_COL = "occupancy_date"
TARGET_COL = "occupancy_pct"

XGB_FEATURE_COLS = [
    "month", "quarter", "day_of_week", "is_weekend", "is_holiday", "is_event",
    "total_rooms",
    "occupancy_pct_lag_1", "occupancy_pct_lag_7", "occupancy_pct_lag_30",
    "occupancy_pct_rolling_mean_7", "occupancy_pct_rolling_mean_30",
]


class OccupancyPipeline(BasePipeline):
    module_name = "occupancy"

    def __init__(self, branch_id: int, start_date: str, end_date: str) -> None:
        super().__init__()
        self.branch_id = branch_id
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self, **kwargs) -> pd.DataFrame:
        from src.database.query import run_query

        sql_path = settings.sql_dir_path / "occupancy.sql"
        return run_query(
            sql_path,
            {
                "branch_id": self.branch_id,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        return handle_missing_values(df, strategy="median")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_calendar_features(df, DATE_COL)
        df = add_lag_features(df, TARGET_COL, lags=[1, 7, 30])
        df = add_rolling_features(df, TARGET_COL, windows=[7, 30])
        df = handle_missing_values(
            df,
            strategy="median",
            columns=[c for c in df.columns if "lag" in c or "rolling" in c],
        )
        return df

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Keep DATE_COL in the feature frame too, so the Prophet model (which
        # ignores XGB_FEATURE_COLS and reads DATE_COL directly) can share the split.
        feature_cols = XGB_FEATURE_COLS + [DATE_COL]
        return default_time_series_split(df, TARGET_COL, feature_cols, test_size=0.2)

    def build_models(self) -> dict[str, BaseMLModel]:
        return {
            "xgboost": OccupancyXGBoostModel(),
            "prophet": OccupancyProphetModel(),
        }

    def train(self, models, X_train, y_train):
        models["xgboost"].feature_names = XGB_FEATURE_COLS
        models["xgboost"].train(X_train[XGB_FEATURE_COLS], y_train)
        models["prophet"].train(X_train[[DATE_COL]], y_train)
        return models

    def evaluate(self, models, X_test, y_test):
        return {
            "xgboost": models["xgboost"].evaluate(X_test[XGB_FEATURE_COLS], y_test),
            "prophet": models["prophet"].evaluate(X_test[[DATE_COL]], y_test),
        }

    def model_save_path(self, model_key: str) -> Path:
        return settings.model_dir_path / f"occupancy_{model_key}.pkl"
