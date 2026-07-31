import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import classification_metrics
from src.models.base import BaseMLModel


class ChurnRandomForestModel(BaseMLModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        preds = self.model.predict(X[self.feature_names])
        proba = self.model.predict_proba(X[self.feature_names])[:, 1]
        return classification_metrics(y.to_numpy(), preds, proba)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]
