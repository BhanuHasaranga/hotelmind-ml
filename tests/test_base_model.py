from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseMLModel


class _DummyModel(BaseMLModel):
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        self.model = {"mean_y": float(y.mean())}

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        preds = self.predict(X)
        return {"mae": float(np.mean(np.abs(preds - y.to_numpy())))}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.model["mean_y"])


def test_save_and_load_round_trip(tmp_path: Path):
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([10, 20, 30])

    model = _DummyModel()
    model.train(X, y)

    save_path = tmp_path / "dummy.pkl"
    model.save(save_path)
    assert save_path.exists()

    loaded = _DummyModel()
    loaded.load(save_path)
    assert loaded.feature_names == ["a"]
    assert loaded.model["mean_y"] == 20.0

    preds = loaded.predict(X)
    assert (preds == 20.0).all()


def test_metrics_helpers_used_correctly():
    from src.evaluation.metrics import classification_metrics, regression_metrics

    reg = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert set(reg.keys()) == {"mae", "rmse", "mape"}
    assert reg["mae"] > 0

    clf = classification_metrics(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([0.1, 0.8, 0.4, 0.2])
    )
    assert set(clf.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
