from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from src.evaluation.metrics import regression_metrics
from src.models.base import BaseMLModel


class OccupancyProphetModel(BaseMLModel):
    """Wraps Prophet for occupancy_pct forecasting.

    Prophet natively produces a confidence interval (yhat_lower/yhat_upper),
    which is why it is used as the primary source for the CI returned by
    predict_occupancy(); XGBoost point forecasts are kept for comparison only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = Prophet(interval_width=0.95, daily_seasonality=False)
        self.date_col: str = "occupancy_date"
        self.target_col: str = "occupancy_pct"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        train_df = pd.DataFrame({"ds": X[self.date_col], "y": y.to_numpy()})
        self.model.fit(train_df)

    def _forecast(self, X: pd.DataFrame) -> pd.DataFrame:
        future = pd.DataFrame({"ds": X[self.date_col]})
        return self.model.predict(future)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        forecast = self._forecast(X)
        return regression_metrics(y.to_numpy(), forecast["yhat"].to_numpy())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._forecast(X)["yhat"].to_numpy()

    def predict_with_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        forecast = self._forecast(X)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={
                "ds": self.date_col,
                "yhat": "predicted_occupancy_pct",
                "yhat_lower": "ci_lower",
                "yhat_upper": "ci_upper",
            }
        )

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model}, path)

    def load(self, path: Path) -> None:
        import joblib

        payload = joblib.load(path)
        self.model = payload["model"]
