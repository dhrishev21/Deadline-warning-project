"""
retrieval.py — Context retrieval for the project assistant.

Aggregates relevant context from all platform modules:
project metrics, forecasts, recommendations, similar projects, simulations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROJECTS_PATH = DATA_DIR / "projects_scored.csv"


def retrieve_project_context(project_id: int) -> Dict:
    """Gather all available context for a specific project."""
    context: Dict = {"project_id": project_id, "sources": []}

    # 1. Core project metrics
    try:
        df = pd.read_csv(PROJECTS_PATH)
        match = df[df["project_id"] == int(project_id)]
        if not match.empty:
            row = match.iloc[0]
            context["metrics"] = {
                "team_size": int(row["team_size"]),
                "sprint_velocity": round(float(row["sprint_velocity"]), 3),
                "tasks_completed_pct": round(float(row["tasks_completed_pct"]), 1),
                "scope_changes": int(row["scope_changes"]),
                "bugs_open": int(row["bugs_open"]),
                "team_availability_pct": round(float(row["team_availability_pct"]), 1),
                "risk_score": round(float(row.get("risk_score", 0)), 4),
                "risk_level": str(row.get("risk_level", "Unknown")),
                "planned_duration_days": int(row["planned_duration_days"]),
                "days_elapsed": int(row["days_elapsed"]),
            }
            if "completion_gap" in row:
                context["metrics"]["completion_gap"] = round(float(row["completion_gap"]), 4)
            context["sources"].append("project_metrics")
    except Exception:
        pass

    # 2. SHAP explanations
    try:
        shap_path = DATA_DIR / "shap_values.json"
        if shap_path.exists():
            shap_data = json.loads(shap_path.read_text(encoding="utf-8"))
            proj_shap = [p for p in shap_data.get("projects", []) if int(p["project_id"]) == int(project_id)]
            if proj_shap:
                drivers = proj_shap[0].get("top_positive", [])[:3]
                reducers = proj_shap[0].get("top_negative", [])[:3]
                context["risk_drivers"] = {
                    "positive": [d.get("description", "") for d in drivers],
                    "negative": [d.get("description", "") for d in reducers],
                }
                context["sources"].append("shap_explanations")
    except Exception:
        pass

    # 3. Recommendations
    try:
        rec_path = DATA_DIR / "recommendations.json"
        if rec_path.exists():
            rec_data = json.loads(rec_path.read_text(encoding="utf-8"))
            proj_recs = [p for p in rec_data.get("projects", []) if int(p["project_id"]) == int(project_id)]
            if proj_recs:
                context["recommendations"] = proj_recs[0].get("recommendations", [])[:4]
                context["sources"].append("recommendation_engine")
    except Exception:
        pass

    # 4. Forecast (if available)
    try:
        from src.forecasting.forecaster import XGBoostForecaster
        forecast = XGBoostForecaster().forecast(project_id, horizon_weeks=4)
        context["forecast"] = {
            "trend": forecast["risk_trend"],
            "weeks": forecast["forecast"][:4],
            "escalation": forecast.get("escalation_warning"),
        }
        context["sources"].append("risk_forecast")
    except Exception:
        pass

    # 5. Similar projects
    try:
        from src.similarity.historical_similarity import find_similar_projects
        similar = find_similar_projects(project_id, top_n=3)
        context["similar_projects"] = {
            "matches": similar["similar_projects"][:3],
            "insight": similar["insight"],
        }
        context["sources"].append("historical_similarity")
    except Exception:
        pass

    return context


def retrieve_portfolio_context() -> Dict:
    """Gather portfolio-level context for broad questions."""
    context: Dict = {"sources": []}

    try:
        df = pd.read_csv(PROJECTS_PATH)
        from src.feature_engineering import engineer_features
        if "completion_gap" not in df.columns:
            df = engineer_features(df)

        context["portfolio"] = {
            "total_projects": len(df),
            "high_risk_count": int((df.get("risk_level", pd.Series()) == "High").sum()),
            "avg_risk": round(float(df.get("risk_score", pd.Series(dtype=float)).mean()) * 100, 1),
            "top_risk_projects": df.nlargest(5, "risk_score")[["project_id", "risk_score", "risk_level"]].to_dict(orient="records") if "risk_score" in df.columns else [],
        }
        context["sources"].append("portfolio_metrics")
    except Exception:
        pass

    return context
