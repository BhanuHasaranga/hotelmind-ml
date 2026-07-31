from pathlib import Path

import pandas as pd

from src.config.constants import MEAL_PERIODS
from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.features.preprocessing import handle_missing_values
from src.features.restaurant_features import derive_meal_quantities
from src.features.time_series_features import add_lag_features, add_rolling_features
from src.models.base import BaseMLModel
from src.models.restaurant.xgboost_model import RestaurantDemandModel
from src.pipelines.base_pipeline import BasePipeline, default_time_series_split

DATE_COL = "date"

FEATURE_COLS = [
    "month", "quarter", "day_of_week", "is_weekend", "is_holiday", "is_event",
    "total_orders_lag_1", "total_orders_lag_7", "total_orders_rolling_mean_7",
]


class RestaurantPipeline(BasePipeline):
    module_name = "restaurant"

    def __init__(self, branch_id: int, start_date: str, end_date: str) -> None:
        super().__init__()
        self.branch_id = branch_id
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self, **kwargs) -> pd.DataFrame:
        from src.database.query import run_query

        sql_path = settings.sql_dir_path / "restaurant.sql"
        return run_query(
            sql_path,
            {
                "branch_id": self.branch_id,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["total_revenue"]).reset_index(drop=True)
        return handle_missing_values(df, strategy="median")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_calendar_features(df, DATE_COL)
        df = derive_meal_quantities(df)
        df = add_lag_features(df, "total_orders", lags=[1, 7])
        df = add_rolling_features(df, "total_orders", windows=[7])
        df = handle_missing_values(
            df, strategy="median", columns=[c for c in df.columns if "lag" in c or "rolling" in c]
        )
        return df

    def split(self, df: pd.DataFrame):
        # split() is per-meal-target; run() is overridden below to loop meals,
        # so this default targets breakfast_revenue only as a placeholder shape.
        return default_time_series_split(df, "breakfast_revenue", FEATURE_COLS, test_size=0.2)

    def build_models(self) -> dict[str, BaseMLModel]:
        return {meal: RestaurantDemandModel() for meal in MEAL_PERIODS}

    def model_save_path(self, model_key: str) -> Path:
        return settings.model_dir_path / f"restaurant_{model_key}.pkl"

    def run(self, **load_kwargs) -> dict[str, dict[str, float]]:
        df = self.load_data(**load_kwargs)
        df = self.clean(df)
        df = self.engineer_features(df)

        models = self.build_models()
        metrics_by_model: dict[str, dict[str, float]] = {}

        for meal, model in models.items():
            target_col = f"{meal}_revenue"
            X_train, X_test, y_train, y_test = default_time_series_split(
                df, target_col, FEATURE_COLS, test_size=0.2
            )
            model.feature_names = FEATURE_COLS
            model.train(X_train, y_train)
            metrics_by_model[meal] = model.evaluate(X_test, y_test)
            model.save(self.model_save_path(meal))
            self.logger.info("Trained+saved meal=%s metrics=%s", meal, metrics_by_model[meal])

        from src.evaluation.report_writer import write_report

        write_report(self.module_name, metrics_by_model)
        return metrics_by_model
