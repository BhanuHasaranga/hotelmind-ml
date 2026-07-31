from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.report_writer import write_report
from src.models.base import BaseMLModel
from src.utils.logging import get_logger


class BasePipeline(ABC):
    """Load -> Clean -> Feature Engineer -> Train -> Evaluate -> Save -> (Predict)

    Domain pipelines override load_data/clean/engineer_features and specify
    which BaseMLModel subclass(es) to train; run() drives the whole sequence
    and writes a metrics report so no module repeats this boilerplate.
    """

    module_name: str

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame: ...

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: ...

    @abstractmethod
    def build_models(self) -> dict[str, BaseMLModel]:
        """Return {model_key: instantiated_but_untrained_model}."""
        ...

    def train(
        self, models: dict[str, BaseMLModel], X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict[str, BaseMLModel]:
        for key, model in models.items():
            self.logger.info("Training model=%s", key)
            model.train(X_train, y_train)
        return models

    def evaluate(
        self, models: dict[str, BaseMLModel], X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, dict[str, float]]:
        metrics_by_model = {}
        for key, model in models.items():
            metrics_by_model[key] = model.evaluate(X_test, y_test)
            self.logger.info("Evaluated model=%s metrics=%s", key, metrics_by_model[key])
        return metrics_by_model

    @abstractmethod
    def model_save_path(self, model_key: str) -> Path: ...

    def save_models(self, models: dict[str, BaseMLModel]) -> None:
        for key, model in models.items():
            path = self.model_save_path(key)
            model.save(path)
            self.logger.info("Saved model=%s to %s", key, path)

    def run(self, **load_kwargs) -> dict[str, dict[str, float]]:
        df = self.load_data(**load_kwargs)
        df = self.clean(df)
        df = self.engineer_features(df)
        X_train, X_test, y_train, y_test = self.split(df)

        models = self.build_models()
        models = self.train(models, X_train, y_train)
        metrics_by_model = self.evaluate(models, X_test, y_test)
        self.save_models(models)

        write_report(self.module_name, metrics_by_model)
        return metrics_by_model


def default_time_series_split(
    df: pd.DataFrame, target_col: str, feature_cols: list[str], test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split (no shuffling) — appropriate for time-series regression."""
    split_idx = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df[target_col],
        test_df[target_col],
    )


def default_random_split(
    df: pd.DataFrame, target_col: str, feature_cols: list[str], test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified-agnostic random split — used for cross-sectional data like churn."""
    return train_test_split(
        df[feature_cols], df[target_col], test_size=test_size, random_state=42
    )
