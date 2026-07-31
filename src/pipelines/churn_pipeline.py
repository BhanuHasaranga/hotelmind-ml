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
        dim_guest = pd.read_parquet(settings.data_warehouse_dir_path / "dim_guest.parquet")
        fact_booking = pd.read_parquet(settings.data_warehouse_dir_path / "fact_booking.parquet")

        total_nights = fact_booking.groupby("guest_key")["nights"].sum().rename("total_nights")
        df = dim_guest.merge(total_nights, left_on="guest_key", right_index=True, how="left")
        df["total_nights"] = df["total_nights"].fillna(0)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["last_stay_date"]).reset_index(drop=True)
        return handle_missing_values(
            df, strategy="median", columns=["lifetime_spend", "total_nights"]
        )

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Anchor the churn snapshot to just after the dataset's own date range
        # rather than the real system clock: the source data only covers
        # 2015-2017, so using today() would make every guest's recency exceed
        # CHURN_WINDOW_DAYS and collapse the label to a single class. See
        # reports/final_phase4/known_limitations.md.
        snapshot_date = (df["last_stay_date"].max() + pd.Timedelta(days=1)).date()
        df = label_churn(df, snapshot_date=snapshot_date)
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
