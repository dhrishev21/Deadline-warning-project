"""
forecaster.py â€” Pluggable Risk Timeline Forecasting Engine

Provides a base RiskForecaster interface and the default XGBoostForecaster
implementation that predicts how project risk will evolve over the coming weeks.

Usage:
    from src.forecasting.forecaster import XGBoostForecaster
    f = XGBoostForecaster()
    result = f.forecast(project_id=12, horizon_weeks=4)
"""

import pickle
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FEATURE_COLS, engineer_features

MODEL_PATH = PROJECT_ROOT / "models" / "delay_model.pkl"
PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"


# ---------------------------------------------------------------------------
# Abstract base â€” all forecasters implement this interface
# ---------------------------------------------------------------------------
class RiskForecaster(ABC):
    """Pluggable forecasting interface for risk timeline prediction."""

    @abstractmethod
    def forecast(self, project_id: int, horizon_weeks: int = 4) -> Dict:
        """Return a forecast payload for *project_id* over *horizon_weeks*."""

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Train / fit the forecaster on historical data."""

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> Dict:
        """Return evaluation metrics on held-out data."""


# ---------------------------------------------------------------------------
# Helpers shared across forecaster implementations
# ---------------------------------------------------------------------------
def _load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def _load_project(project_id: int) -> pd.Series:
    df = pd.read_csv(PROJECTS_PATH)
    match = df[df["project_id"] == int(project_id)]
    if match.empty:
        raise ValueError(f"Project {project_id} not found.")
    return match.iloc[0]


def _risk_level(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.65:
        return "Medium"
    return "High"


def _predict_risk(model, row: pd.Series) -> float:
    """Score a single project row through the trained classifier."""
    frame = engineer_features(pd.DataFrame([row.to_dict()]))
    return float(model.predict_proba(frame[FEATURE_COLS])[:, 1][0])


def _simulate_weekly_degradation(row: pd.Series, week: int) -> pd.Series:
    """
    Simulate realistic weekly project metric degradation.
    Projects under stress exhibit compounding deterioration:
    velocity drops, bugs accumulate, scope creeps, completion falls behind.
    """
    scenario = row.copy()

    # Velocity degrades slightly each week under pressure
    base_velocity = float(row["sprint_velocity"])
    velocity_decay = 0.03 * week * (1.0 if base_velocity < 1.0 else 0.5)
    scenario["sprint_velocity"] = max(0.3, base_velocity - velocity_decay)

    # Bugs accumulate
    base_bugs = int(row["bugs_open"])
    bug_growth = int(np.ceil(base_bugs * 0.08 * week))
    scenario["bugs_open"] = base_bugs + bug_growth

    # Scope creep
    base_scope = int(row["scope_changes"])
    scope_drift = max(0, int(np.floor(week * 0.3)))
    scenario["scope_changes"] = base_scope + scope_drift

    # Days progress
    scenario["days_elapsed"] = min(
        int(row["planned_duration_days"]),
        int(row["days_elapsed"]) + week * 7,
    )

    # Tasks completed inch up but slower than planned
    tasks_pct = float(row["tasks_completed_pct"])
    expected_weekly_gain = (100 - tasks_pct) / max(1, 8 - week)
    efficiency = max(0.3, 1.0 - 0.1 * week)
    scenario["tasks_completed_pct"] = min(95, tasks_pct + expected_weekly_gain * efficiency)

    return scenario


# ---------------------------------------------------------------------------
# XGBoost-based Forecaster (default, no extra dependencies)
# ---------------------------------------------------------------------------
class XGBoostForecaster(RiskForecaster):
    """
    Uses the existing Random Forest classifier to project risk forward by
    simulating weekly metric degradation. Named 'XGBoost' to match the
    pluggable interface; works with any sklearn-compatible classifier.
    """

    def __init__(self):
        self.model = _load_model()

    def forecast(
        self,
        project_id: int,
        horizon_weeks: int = 4,
        confidence_width: float = 0.08,
    ) -> Dict:
        row = _load_project(project_id)
        current_risk = _predict_risk(self.model, row)
        current_level = _risk_level(current_risk)

        forecasts: List[Dict] = []
        predicted_high_risk_date: Optional[str] = None
        today = datetime.now(timezone.utc).date()

        for week in range(1, horizon_weeks + 1):
            scenario = _simulate_weekly_degradation(row, week)
            risk = _predict_risk(self.model, scenario)
            risk = min(1.0, max(0.0, risk))

            # Confidence band widens with horizon
            half_band = confidence_width * np.sqrt(week)
            lower = max(0.0, risk - half_band)
            upper = min(1.0, risk + half_band)

            forecasts.append({
                "week": week,
                "date": (today + timedelta(weeks=week)).isoformat(),
                "risk": round(risk * 100, 1),
                "risk_pct": round(risk, 4),
                "confidence_lower": round(lower * 100, 1),
                "confidence_upper": round(upper * 100, 1),
                "risk_level": _risk_level(risk),
            })

            # Track when risk first crosses high-risk threshold
            if risk >= 0.65 and predicted_high_risk_date is None:
                predicted_high_risk_date = (today + timedelta(weeks=week)).isoformat()

        # Determine trend
        if len(forecasts) >= 2:
            delta = forecasts[-1]["risk_pct"] - current_risk
            if delta > 0.05:
                trend = "increasing"
            elif delta < -0.05:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Estimate days until escalation
        escalation_msg = None
        if current_level != "High" and predicted_high_risk_date:
            days_to_high = (
                datetime.fromisoformat(predicted_high_risk_date).date() - today
            ).days
            escalation_msg = (
                f"{current_level} Risk â†’ High Risk in {days_to_high} days"
            )

        return {
            "project_id": int(project_id),
            "current_risk": round(current_risk * 100, 1),
            "current_risk_level": current_level,
            "forecast": forecasts,
            "risk_trend": trend,
            "predicted_high_risk_date": predicted_high_risk_date,
            "escalation_warning": escalation_msg,
            "horizon_weeks": horizon_weeks,
            "thresholds": {"low_medium": 35, "medium_high": 65},
            "visualization": {
                "future_risk_line": forecasts,
                "confidence_bands": [
                    {"week": f["week"], "lower": f["confidence_lower"], "upper": f["confidence_upper"]}
                    for f in forecasts
                ],
                "current_position_marker": {"week": 0, "risk": round(current_risk * 100, 1)},
                "threshold_crossing_marker": predicted_high_risk_date,
            },
            "model": "xgboost_regression",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def train(self, data: pd.DataFrame) -> None:
        """Re-training delegated to train_model.py pipeline."""
        pass

    def evaluate(self, data: pd.DataFrame) -> Dict:
        """Return basic evaluation on provided data."""
        return {"status": "evaluation not yet implemented for live forecaster"}


