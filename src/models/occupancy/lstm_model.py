import numpy as np
import pandas as pd

from src.models.base import BaseMLModel


class OccupancyLSTMModel(BaseMLModel):
    """Scaffold only — not wired into the pipeline or prediction API.

    TODO: implement sequence-batching utilities (sliding windows over
    occupancy_pct per branch) and a torch/keras LSTM once the project
    is ready to invest in a deep-learning forecasting path.
    """

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        # TODO: build sliding-window sequences and train an LSTM (torch or keras)
        raise NotImplementedError("LSTM occupancy model is not yet implemented")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        # TODO: evaluate using regression_metrics once train() is implemented
        raise NotImplementedError("LSTM occupancy model is not yet implemented")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # TODO: run inference over sliding windows once train() is implemented
        raise NotImplementedError("LSTM occupancy model is not yet implemented")
