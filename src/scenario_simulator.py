"""
scenario_simulator.py
Runs what-if project risk simulations and returns JSON-ready comparison payloads.
"""

import pickle
from pathlib import Path
import sys
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import explain_project_frame

from src.feature_engineering import FEATURE_COLS, engineer_features

from src.recommendation_engine import generate_project_recommendations, risk_level


MODEL_PATH = PROJECT_ROOT / "models" / "delay_model.pkl"
PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"


def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def load_project(project_id: int) -> pd.Series:
    df = pd.read_csv(PROJECTS_PATH)
    match = df[df["project_id"] == int(project_id)]
    if match.empty:
        raise ValueError(f"Project {project_id} was not found.")
    return match.iloc[0]


def predict_project(model, row: pd.Series) -> Dict[str, object]:
    frame = engineer_features(pd.DataFrame([row.to_dict()]))
    score = float(model.predict_proba(frame[FEATURE_COLS])[:, 1][0])
    return {"risk_score": score, "risk_level": risk_level(score)}


def apply_overrides(row: pd.Series, overrides: Dict[str, object]) -> pd.Series:
    scenario = row.copy()

    field_map = {
        "team_size": "team_size",
        "sprint_velocity": "sprint_velocity",
        "project_scope": "scope_changes",
        "scope": "scope_changes",
        "scope_changes": "scope_changes",
        "bug_count": "bugs_open",
        "bugs_open": "bugs_open",
        "team_availability_pct": "team_availability_pct",
        "tasks_completed_pct": "tasks_completed_pct",
    }

    for incoming_key, row_key in field_map.items():
        if incoming_key in overrides and overrides[incoming_key] is not None:
            scenario[row_key] = overrides[incoming_key]

    if overrides.get("deadline_extension_days") is not None:
        scenario["planned_duration_days"] = float(row["planned_duration_days"]) + float(overrides["deadline_extension_days"])

    return scenario


def simulate(project_id: int, overrides: Dict[str, object]) -> Dict[str, object]:
    model = load_model()
    current_row = load_project(project_id)
    scenario_row = apply_overrides(current_row, overrides)

    current_prediction = predict_project(model, current_row)
    scenario_prediction = predict_project(model, scenario_row)
    improvement = current_prediction["risk_score"] - scenario_prediction["risk_score"]

    current_explanation = explain_project_frame(pd.DataFrame([current_row.to_dict()]), model)
    scenario_explanation = explain_project_frame(pd.DataFrame([scenario_row.to_dict()]), model)
    recommendations = generate_project_recommendations(scenario_row, model)

    return {
        "project_id": int(project_id),
        "current": {
            "features": _public_features(current_row),
            **current_prediction,
            "explanation": current_explanation,
        },
        "scenario": {
            "features": _public_features(scenario_row),
            **scenario_prediction,
            "explanation": scenario_explanation,
            "recommendations": recommendations["recommendations"],
        },
        "difference": {
            "risk_delta": scenario_prediction["risk_score"] - current_prediction["risk_score"],
            "risk_reduction": max(0.0, improvement),
            "improvement_pct": max(0.0, improvement) * 100,
        },
    }


def _public_features(row: pd.Series) -> Dict[str, float]:
    return {
        "team_size": int(row["team_size"]),
        "sprint_velocity": float(row["sprint_velocity"]),
        "scope_changes": int(row["scope_changes"]),
        "bugs_open": int(row["bugs_open"]),
        "planned_duration_days": int(row["planned_duration_days"]),
        "tasks_completed_pct": float(row["tasks_completed_pct"]),
        "team_availability_pct": float(row["team_availability_pct"]),
    }


