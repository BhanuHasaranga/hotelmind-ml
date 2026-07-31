from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.features.churn_features import add_rfm_features, label_churn
from src.features.preprocessing import handle_missing_values
from src.models.base import BaseMLModel
from src.models.churn.random_forest_model import ChurnRandomForestModel
from src.models.churn.xgboost_model import ChurnXGBoostModel
from src.pipelines.base_pipeline import BasePipeline, default_random_split

TARGET_COL = "churn"
FEATURE_COLS = ["recency_days", "frequency", "monetary", "avg_spend_per_stay", "total_nights"]


class ChurnPipeline(BasePipeline):
    module_name = "churn"

    def load_data(self, **kwargs) -> pd.DataFrame:
        from src.database.query import run_query

        sql_path = settings.sql_dir_path / "guest.sql"
        return run_query(sql_path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["last_stay_date"]).reset_index(drop=True)
        return handle_missing_values(
            df, strategy="median", columns=["lifetime_spend", "total_nights", "avg_spend_per_stay"]
        )

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = label_churn(df)
        df = add_rfm_features(df)
        return df

    def split(self, df: pd.DataFrame):
        return default_random_split(df, TARGET_COL, FEATURE_COLS, test_size=0.2)

    def build_models(self) -> dict[str, BaseMLModel]:
        return {
            "random_forest": ChurnRandomForestModel(),
            "xgboost": ChurnXGBoostModel(),
        }

    def model_save_path(self, model_key: str) -> Path:
        return settings.model_dir_path / f"churn_{model_key}.pkl"
