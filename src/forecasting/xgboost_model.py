"""
xgboost_model.py - Optional XGBoost-style risk forecaster.

The platform exposes this class even when xgboost is not installed. In that case it
falls back to the production-safe sklearn model projection used by XGBoostForecaster.
"""

from typing import Dict

try:
    from xgboost import XGBRegressor  # noqa: F401
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from src.forecasting.forecaster import XGBoostForecaster


class XGBoostRiskForecaster(XGBoostForecaster):
    """XGBoost-compatible forecaster wrapper with graceful fallback."""

    def forecast(self, project_id: int, horizon_weeks: int = 4, confidence_width: float = 0.08) -> Dict:
        result = super().forecast(project_id, horizon_weeks, confidence_width)
        result["model"] = "xgboost_regression" if XGBOOST_AVAILABLE else "xgboost_interface_sklearn_fallback"
        result["xgboost_available"] = XGBOOST_AVAILABLE
        return result
