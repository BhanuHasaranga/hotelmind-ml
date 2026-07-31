from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.features.occupancy_aggregation import load_daily_occupancy
from src.features.preprocessing import encode_categoricals, handle_missing_values
from src.features.time_series_features import add_lag_features, add_rolling_features
from src.pipelines.synthetic_data import ensure_staffing_seed
from src.models.base import BaseMLModel
from src.models.staffing.regression_model import StaffingRegressionModel
from src.pipelines.base_pipeline import BasePipeline, default_time_series_split

DATE_COL = "date"
TARGET_COL = "present_employees"

CATEGORICAL_COLS = ["department_name"]
NUMERIC_FEATURE_COLS = [
    "month", "quarter", "day_of_week", "is_weekend", "is_holiday",
    "scheduled_employees",
    "present_employees_lag_7", "present_employees_rolling_mean_7",
]
FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_COLS


class StaffingPipeline(BasePipeline):
    module_name = "staffing"

    def __init__(self, branch_id: int, start_date: str, end_date: str) -> None:
        super().__init__()
        self.branch_id = branch_id
        self.start_date = start_date
        self.end_date = end_date
        self.encoder = None

    def load_data(self, **kwargs) -> pd.DataFrame:
        daily_occupancy = load_daily_occupancy()  # unfiltered: seed covers all branches
        seed_path = ensure_staffing_seed(daily_occupancy)
        df = pd.read_csv(seed_path, parse_dates=[DATE_COL])

        df = df[df["branch_id"] == self.branch_id].reset_index(drop=True)
        mask = (df[DATE_COL] >= self.start_date) & (df[DATE_COL] <= self.end_date)
        return df[mask].reset_index(drop=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        return handle_missing_values(df, strategy="median")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["department_id", DATE_COL]).reset_index(drop=True)
        df = add_calendar_features(df, DATE_COL)
        df = add_lag_features(df, TARGET_COL, lags=[7], group_col="department_id")
        df = add_rolling_features(df, TARGET_COL, windows=[7], group_col="department_id")
        df = handle_missing_values(
            df, strategy="median", columns=[c for c in df.columns if "lag" in c or "rolling" in c]
        )
        df, self.encoder = encode_categoricals(df, CATEGORICAL_COLS)
        return df

    def split(self, df: pd.DataFrame):
        return default_time_series_split(df, TARGET_COL, FEATURE_COLS, test_size=0.2)

    def build_models(self) -> dict[str, BaseMLModel]:
        return {"regression": StaffingRegressionModel()}

    def train(self, models, X_train, y_train):
        models["regression"].encoders = {"categorical": self.encoder}
        models["regression"].feature_names = FEATURE_COLS
        models["regression"].train(X_train, y_train)
        return models

    def model_save_path(self, model_key: str) -> Path:
        return settings.model_dir_path / f"staffing_{model_key}.pkl"
