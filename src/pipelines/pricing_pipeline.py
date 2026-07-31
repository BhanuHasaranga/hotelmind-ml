from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.features.occupancy_aggregation import load_daily_occupancy
from src.features.preprocessing import (
    encode_categoricals,
    handle_missing_values,
    scale_features,
)
from src.features.pricing_features import add_demand_index, add_room_types
from src.models.base import BaseMLModel
from src.models.pricing.xgboost_model import PricingXGBoostModel
from src.pipelines.base_pipeline import BasePipeline, default_time_series_split

DATE_COL = "date"
TARGET_COL = "avg_daily_rate"

CATEGORICAL_COLS = ["season", "room_type_name"]
NUMERIC_FEATURE_COLS = [
    "occupancy_pct", "occupancy_7day_avg", "demand_index",
    "is_weekend", "is_holiday", "is_event", "base_price_multiplier",
]
FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_COLS


class PricingPipeline(BasePipeline):
    module_name = "pricing"

    def __init__(self, branch_id: int, start_date: str, end_date: str) -> None:
        super().__init__()
        self.branch_id = branch_id
        self.start_date = start_date
        self.end_date = end_date
        self.encoder = None
        self.scaler = None

    def load_data(self, **kwargs) -> pd.DataFrame:
        df = load_daily_occupancy(branch_id=self.branch_id)
        df = df.rename(columns={"occupancy_date": DATE_COL})
        mask = (df[DATE_COL] >= self.start_date) & (df[DATE_COL] <= self.end_date)
        return df[mask].reset_index(drop=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        return handle_missing_values(df, strategy="median")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_calendar_features(df, DATE_COL)
        df = add_demand_index(df)
        df = add_room_types(df)
        df = handle_missing_values(df, strategy="median", columns=NUMERIC_FEATURE_COLS)
        df, self.encoder = encode_categoricals(df, CATEGORICAL_COLS)
        df, self.scaler = scale_features(df, NUMERIC_FEATURE_COLS)
        return df

    def split(self, df: pd.DataFrame):
        return default_time_series_split(df, TARGET_COL, FEATURE_COLS, test_size=0.2)

    def build_models(self) -> dict[str, BaseMLModel]:
        return {"xgboost": PricingXGBoostModel()}

    def train(self, models, X_train, y_train):
        models["xgboost"].encoders = {"categorical": self.encoder}
        models["xgboost"].scaler = self.scaler
        models["xgboost"].feature_names = FEATURE_COLS
        models["xgboost"].train(X_train, y_train)
        return models

    def model_save_path(self, model_key: str) -> Path:
        return settings.model_dir_path / f"pricing_{model_key}.pkl"
