from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseMLModel(ABC):
    """Common interface every domain model (occupancy, pricing, restaurant,
    staffing, churn) implements, so pipelines can train/evaluate/save/load
    any algorithm identically.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.encoders: dict[str, Any] = {}
        self.scaler: Any = None
        self.feature_names: list[str] = []

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def save(self, path: Path) -> None:
        payload = {
            "model": self.model,
            "encoders": self.encoders,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self.encoders = payload.get("encoders", {})
        self.scaler = payload.get("scaler")
        self.feature_names = payload.get("feature_names", [])
