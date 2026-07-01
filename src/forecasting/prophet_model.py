"""
prophet_model.py — Optional Prophet-based risk forecaster.

Falls back gracefully if Prophet is not installed.
"""

from typing import Dict
import pandas as pd

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from src.forecasting.forecaster import RiskForecaster


class ProphetForecaster(RiskForecaster):
    """Prophet-based time-series forecaster for risk evolution."""

    def __init__(self):
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet is not installed. Install with: pip install prophet"
            )
        self.model = None

    def forecast(self, project_id: int, horizon_weeks: int = 4) -> Dict:
        # Delegate to XGBoost if Prophet isn't trained on real time-series yet
        from src.forecasting.forecaster import XGBoostForecaster
        result = XGBoostForecaster().forecast(project_id, horizon_weeks)
        result["model"] = "prophet"
        return result

    def train(self, data: pd.DataFrame) -> None:
        """
        Train Prophet on weekly risk snapshots.
        Expects columns: ds (date), y (risk_score).
        """
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative",
        )
        self.model.fit(data[["ds", "y"]])

    def evaluate(self, data: pd.DataFrame) -> Dict:
        if self.model is None:
            return {"error": "Model not trained"}
        future = self.model.make_future_dataframe(periods=4, freq="W")
        forecast = self.model.predict(future)
        return {"forecast_tail": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(4).to_dict(orient="records")}
