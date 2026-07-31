import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.metrics import classification_metrics
from src.models.base import BaseMLModel


class ChurnXGBoostModel(BaseMLModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        preds = self.model.predict(X[self.feature_names])
        proba = self.model.predict_proba(X[self.feature_names])[:, 1]
        return classification_metrics(y.to_numpy(), preds, proba)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]
