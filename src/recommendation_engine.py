"""
recommendation_engine.py
Creates ranked prescriptive actions by simulating counterfactual project scenarios.
Run: python src/recommendation_engine.py
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FEATURE_COLS, engineer_features


MODEL_PATH = PROJECT_ROOT / "models" / "delay_model.pkl"
PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"
RECOMMENDATIONS_PATH = PROJECT_ROOT / "data" / "recommendations.json"


DIFFICULTY_WEIGHTS = {"Low": 1.0, "Medium": 1.5, "High": 2.2}


def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def predict_risk(model, project_row: pd.Series) -> float:
    frame = pd.DataFrame([project_row.to_dict()])
    frame = engineer_features(frame)
    return float(model.predict_proba(frame[FEATURE_COLS])[:, 1][0])


def risk_level(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.65:
        return "Medium"
    return "High"


def _with_action(row: pd.Series, updates: Dict[str, float]) -> pd.Series:
    scenario = row.copy()
    for key, value in updates.items():
        scenario[key] = value
    return scenario


def candidate_actions(row: pd.Series) -> List[Dict[str, object]]:
    actions = []

    if row["team_size"] < 20:
        target_team = int(min(20, max(row["team_size"] + 2, 5)))
        actions.append(
            {
                "recommendation": f"Increase team size from {int(row['team_size'])} to {target_team}",
                "updates": {"team_size": target_team},
                "implementation_difficulty": "High",
                "action_type": "staffing",
            }
        )

    if row["scope_changes"] > 0:
        target_scope = int(max(0, np.floor(row["scope_changes"] * 0.8)))
        actions.append(
            {
                "recommendation": "Reduce scope by 20%",
                "updates": {"scope_changes": target_scope},
                "implementation_difficulty": "Medium",
                "action_type": "scope_control",
            }
        )

    actions.append(
        {
            "recommendation": "Extend deadline by 2 weeks",
            "updates": {"planned_duration_days": int(row["planned_duration_days"] + 14)},
            "implementation_difficulty": "Medium",
            "action_type": "timeline",
        }
    )

    if row["sprint_velocity"] < 1.5:
        target_velocity = round(float(min(1.5, max(row["sprint_velocity"] * 1.2, row["sprint_velocity"] + 0.15))), 2)
        actions.append(
            {
                "recommendation": f"Improve sprint velocity from {row['sprint_velocity']:.2f} to {target_velocity:.2f}",
                "updates": {"sprint_velocity": target_velocity},
                "implementation_difficulty": "Medium",
                "action_type": "delivery_flow",
            }
        )

    if row["bugs_open"] > 0:
        target_bugs = int(max(0, np.floor(row["bugs_open"] * 0.7)))
        actions.append(
            {
                "recommendation": f"Reduce open bugs from {int(row['bugs_open'])} to {target_bugs}",
                "updates": {"bugs_open": target_bugs},
                "implementation_difficulty": "Low",
                "action_type": "quality",
            }
        )

    if row["team_availability_pct"] < 95:
        target_availability = round(float(min(98, max(90, row["team_availability_pct"] + 10))), 1)
        actions.append(
            {
                "recommendation": f"Raise team availability to {target_availability:.0f}%",
                "updates": {"team_availability_pct": target_availability},
                "implementation_difficulty": "Low",
                "action_type": "capacity",
            }
        )

    return actions


def generate_project_recommendations(row: pd.Series, model=None, max_items: int = 4) -> Dict[str, object]:
    model = model or load_model()
    current_risk = float(row.get("risk_score", predict_risk(model, row)))
    ranked = []

    for action in candidate_actions(row):
        scenario = _with_action(row, action["updates"])
        new_risk = predict_risk(model, scenario)
        risk_reduction = max(0.0, current_risk - new_risk)
        difficulty_weight = DIFFICULTY_WEIGHTS[action["implementation_difficulty"]]
        priority_score = (risk_reduction * 100) / difficulty_weight
        ranked.append(
            {
                "recommendation": action["recommendation"],
                "expected_new_risk": new_risk,
                "expected_risk_reduction": risk_reduction,
                "estimated_improvement_pct": risk_reduction * 100,
                "implementation_difficulty": action["implementation_difficulty"],
                "priority_score": priority_score,
                "action_type": action["action_type"],
                "intervention": action["updates"],
            }
        )

    ranked = sorted(ranked, key=lambda item: item["priority_score"], reverse=True)
    project_id = int(row["project_id"]) if "project_id" in row else None
    return {
        "project_id": project_id,
        "current_risk": current_risk,
        "current_risk_level": risk_level(current_risk),
        "recommendations": ranked[:max_items],
    }


def generate_recommendations() -> Dict[str, object]:
    model = load_model()
    df = pd.read_csv(PROJECTS_PATH)
    if any(col not in df.columns for col in FEATURE_COLS):
        df = engineer_features(df)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "projects": [generate_project_recommendations(row, model) for _, row in df.iterrows()],
    }
    RECOMMENDATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECOMMENDATIONS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Recommendations saved -> {RECOMMENDATIONS_PATH.relative_to(PROJECT_ROOT)}")
    return payload


if __name__ == "__main__":
    generate_recommendations()


