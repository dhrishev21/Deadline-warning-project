"""
explainability.py
Generates per-project and portfolio-level SHAP explanations for risk predictions.
Run: python src/explainability.py
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FEATURE_COLS, engineer_features


MODEL_PATH = PROJECT_ROOT / "models" / "delay_model.pkl"
SCORED_PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"
SHAP_JSON_PATH = PROJECT_ROOT / "data" / "shap_values.json"
SHAP_CHART_PATH = PROJECT_ROOT / "charts" / "shap_importance.png"


FEATURE_LABELS = {
    "team_size": "Team size",
    "sprint_velocity": "Sprint velocity",
    "tasks_completed_pct": "Task completion",
    "scope_changes": "Scope changes",
    "bugs_open": "Open bugs",
    "team_availability_pct": "Team availability",
    "velocity_trend": "Velocity trend",
    "scope_risk": "Scope and bug compound risk",
    "completion_gap": "Completion gap",
    "bug_density": "Bug density",
}


def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def load_projects(path: Path = SCORED_PROJECTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        df = engineer_features(df)
    return df


def is_tree_model(model) -> bool:
    return hasattr(model, "estimators_") or hasattr(model, "tree_") or hasattr(model, "get_booster")


def _class_one_shap_values(raw_values):
    """Normalize SHAP return shapes across SHAP versions and binary classifiers."""
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])

    raw_values = np.asarray(raw_values)
    if raw_values.ndim == 3:
        return raw_values[:, :, 1]
    return raw_values


def _class_one_expected_value(expected_value):
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        value = np.asarray(expected_value)
        if value.ndim > 0 and len(value) > 1:
            return float(value[1])
        return float(value.ravel()[0])
    return float(expected_value)


def compute_shap_values(model, feature_frame: pd.DataFrame) -> Dict[str, object]:
    """
    Use SHAP TreeExplainer for tree-based models.
    Falls back to a deterministic feature-importance approximation when SHAP is not installed,
    keeping the dashboard usable while requirements.txt documents the production dependency.
    """
    if is_tree_model(model):
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(feature_frame)
            return {
                "method": "shap_tree_explainer",
                "values": _class_one_shap_values(raw_values),
                "base_value": _class_one_expected_value(explainer.expected_value),
            }
        except ImportError:
            pass

    # Fallback keeps generated demo artifacts available in environments without SHAP.
    probabilities = model.predict_proba(feature_frame)[:, 1]
    baseline = float(probabilities.mean())
    importances = np.asarray(getattr(model, "feature_importances_", np.ones(len(FEATURE_COLS))))
    importances = importances / importances.sum()
    standardized = (feature_frame - feature_frame.mean()) / feature_frame.std(ddof=0).replace(0, 1)
    approx_values = standardized.to_numpy() * importances
    row_sums = approx_values.sum(axis=1)
    scale = np.divide(
        probabilities - baseline,
        row_sums,
        out=np.zeros_like(probabilities),
        where=np.abs(row_sums) > 1e-9,
    )
    approx_values = approx_values * scale[:, None]
    return {
        "method": "feature_importance_approximation",
        "values": approx_values,
        "base_value": baseline,
    }


def contributor_description(feature: str, impact: float, row: pd.Series) -> str:
    sign = "contributed" if impact >= 0 else "reduced risk by"
    pct = abs(impact) * 100

    if feature == "scope_changes":
        driver = "Scope increase"
    elif feature == "scope_risk":
        driver = "Scope and bug pressure"
    elif feature in {"sprint_velocity", "velocity_trend"}:
        driver = "Velocity decrease" if impact >= 0 else "Healthy velocity"
    elif feature == "team_size":
        driver = "Small team size" if row.get("team_size", 0) < 6 else "Team capacity"
    elif feature == "team_availability_pct":
        driver = "Low availability" if impact >= 0 else "Experienced team availability"
    elif feature in {"bugs_open", "bug_density"}:
        driver = "Bug load"
    elif feature == "completion_gap":
        driver = "Schedule gap" if row.get("completion_gap", 0) < 0 else "Schedule progress"
    elif feature == "tasks_completed_pct":
        driver = "Incomplete work" if impact >= 0 else "Task completion"
    else:
        driver = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

    if impact >= 0:
        return f"{driver} contributed +{pct:.1f}%"
    return f"{driver} reduced risk by -{pct:.1f}%"


def build_project_explanations(df: pd.DataFrame, shap_values: np.ndarray, base_value: float) -> List[Dict[str, object]]:
    projects = []
    for idx, row in df.reset_index(drop=True).iterrows():
        contributors = []
        for feature, shap_value in zip(FEATURE_COLS, shap_values[idx]):
            impact = float(shap_value)
            contributors.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                    "value": float(row[feature]),
                    "shap_value": impact,
                    "impact_pct": impact * 100,
                    "direction": "increases_risk" if impact >= 0 else "reduces_risk",
                    "description": contributor_description(feature, impact, row),
                }
            )

        contributors = sorted(contributors, key=lambda c: abs(c["shap_value"]), reverse=True)
        positive = [c for c in contributors if c["shap_value"] > 0]
        negative = [c for c in contributors if c["shap_value"] < 0]
        project_id = int(row["project_id"]) if "project_id" in row else int(idx + 1)
        projects.append(
            {
                "project_id": project_id,
                "risk_score": float(row.get("risk_score", np.nan)),
                "risk_level": str(row.get("risk_level", "")),
                "base_value": base_value,
                "top_positive": positive[:5],
                "top_negative": negative[:5],
                "contributors": contributors,
            }
        )
    return projects


def build_portfolio_explanation(projects: Iterable[Dict[str, object]], project_ids: Optional[Iterable[int]] = None) -> Dict[str, object]:
    project_list = list(projects)
    if project_ids is not None:
        selected_ids = {int(project_id) for project_id in project_ids}
        project_list = [p for p in project_list if int(p["project_id"]) in selected_ids]

    if not project_list:
        return {"project_count": 0, "average_abs_impact": [], "positive_drivers": [], "negative_drivers": []}

    rows = []
    for project in project_list:
        for contributor in project["contributors"]:
            rows.append(contributor)

    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["feature", "label"], as_index=False).agg(
        mean_impact=("shap_value", "mean"),
        mean_abs_impact=("shap_value", lambda s: float(np.mean(np.abs(s)))),
    )
    grouped["impact_pct"] = grouped["mean_impact"] * 100
    grouped["abs_impact_pct"] = grouped["mean_abs_impact"] * 100
    grouped = grouped.sort_values("mean_abs_impact", ascending=False)

    positive = grouped[grouped["mean_impact"] > 0].sort_values("mean_impact", ascending=False)
    negative = grouped[grouped["mean_impact"] < 0].sort_values("mean_impact", ascending=True)

    return {
        "project_count": len(project_list),
        "average_abs_impact": grouped.head(10).to_dict(orient="records"),
        "positive_drivers": positive.head(5).to_dict(orient="records"),
        "negative_drivers": negative.head(5).to_dict(orient="records"),
    }


def save_shap_chart(portfolio: Dict[str, object], output_path: Path = SHAP_CHART_PATH) -> None:
    data = portfolio.get("average_abs_impact", [])[:8]
    if not data:
        return

    labels = [item["label"] for item in data][::-1]
    values = [item["abs_impact_pct"] for item in data][::-1]
    colors = ["#E24B4A" if i >= len(values) - 3 else "#378ADD" for i in range(len(values))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color=colors)
    plt.title("SHAP Risk Contribution Importance", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Mean absolute contribution to risk score (percentage points)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_explanations() -> Dict[str, object]:
    model = load_model()
    df = load_projects()
    feature_frame = df[FEATURE_COLS]

    shap_result = compute_shap_values(model, feature_frame)
    projects = build_project_explanations(df, shap_result["values"], shap_result["base_value"])
    portfolio = build_portfolio_explanation(projects)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": shap_result["method"],
        "model_type": type(model).__name__,
        "features": FEATURE_COLS,
        "projects": projects,
        "portfolio": portfolio,
    }

    SHAP_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHAP_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_shap_chart(portfolio)
    print(f"Explanations saved -> {SHAP_JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"SHAP chart saved -> {SHAP_CHART_PATH.relative_to(PROJECT_ROOT)}")
    return payload


def explain_project_frame(project_frame: pd.DataFrame, model=None) -> Dict[str, object]:
    """Return explanation payload for ad-hoc single/scenario prediction APIs."""
    model = model or load_model()
    engineered = engineer_features(project_frame)
    feature_frame = engineered[FEATURE_COLS]
    risk_score = model.predict_proba(feature_frame)[:, 1]
    engineered = engineered.copy()
    engineered["risk_score"] = risk_score
    engineered["risk_level"] = pd.cut(
        engineered["risk_score"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    shap_result = compute_shap_values(model, feature_frame)
    return build_project_explanations(engineered, shap_result["values"], shap_result["base_value"])[0]


if __name__ == "__main__":
    generate_explanations()


