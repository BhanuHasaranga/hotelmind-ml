import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.evaluation.metrics import regression_metrics
from src.models.base import BaseMLModel


class OccupancyXGBoostModel(BaseMLModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        preds = self.predict(X)
        return regression_metrics(y.to_numpy(), preds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])
