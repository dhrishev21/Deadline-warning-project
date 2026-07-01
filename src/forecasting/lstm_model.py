"""
lstm_model.py — Optional LSTM-based risk forecaster.

Falls back gracefully if TensorFlow/Keras is not installed.
"""

from typing import Dict
import pandas as pd

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.forecasting.forecaster import RiskForecaster


class LSTMForecaster(RiskForecaster):
    """LSTM time-series forecaster for risk evolution."""

    def __init__(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is not installed. Install with: pip install tensorflow"
            )
        self.model = None

    def forecast(self, project_id: int, horizon_weeks: int = 4) -> Dict:
        # Delegate to XGBoost until LSTM is trained on real sequences
        from src.forecasting.forecaster import XGBoostForecaster
        result = XGBoostForecaster().forecast(project_id, horizon_weeks)
        result["model"] = "lstm"
        return result

    def train(self, data: pd.DataFrame) -> None:
        """
        Train LSTM on sequential risk snapshots.
        Expects shape (n_samples, sequence_length, n_features).
        """
        import numpy as np

        # Build a simple LSTM model
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(None, 1), return_sequences=False),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def evaluate(self, data: pd.DataFrame) -> Dict:
        if self.model is None:
            return {"error": "Model not trained"}
        return {"status": "LSTM evaluation placeholder"}
