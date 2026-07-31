import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.evaluation.metrics import regression_metrics
from src.models.base import BaseMLModel


class StaffingRegressionModel(BaseMLModel):
    """Single shared regression model across departments (Reception, Kitchen,
    Housekeeping), with department + branch encoded as categorical features
    (departments share the same schema/grain in mart_staff_daily, unlike
    restaurant meals, so one model generalizes better than three).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        preds = self.predict(X)
        return regression_metrics(y.to_numpy(), preds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])
