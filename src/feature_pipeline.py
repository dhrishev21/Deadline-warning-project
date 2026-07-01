"""
feature_pipeline.py
Normalizes GitHub/Jira metrics into model-ready project features.
Run demo normalization: python src/feature_pipeline.py
"""

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FEATURE_COLS, engineer_features

from src.github_connector import GitHubConnector

from src.jira_connector import JiraConnector


MODEL_PATH = PROJECT_ROOT / "models" / "delay_model.pkl"
ENGINEERING_METRICS_PATH = PROJECT_ROOT / "data" / "engineering_metrics.json"
NORMALIZED_FEATURES_PATH = PROJECT_ROOT / "data" / "normalized_engineering_features.json"


def normalize_metrics(
    github_metrics: Optional[Dict[str, object]] = None,
    jira_metrics: Optional[Dict[str, object]] = None,
    project_id: int = 1001,
    project_name: str = "Integrated Engineering Project",
) -> Dict[str, object]:
    github_metrics = github_metrics or {}
    jira_metrics = jira_metrics or {}

    contributors = int(github_metrics.get("number_of_contributors") or 6)
    commits = int(github_metrics.get("commits_per_week") or 12)
    open_prs = int(github_metrics.get("pull_requests_opened") or 3)
    merge_hours = float(github_metrics.get("average_merge_time_hours") or 18)
    blocked = int(jira_metrics.get("blocked_issues") or 1)
    bug_count = int(jira_metrics.get("bug_count") or github_metrics.get("issue_creation_rate") or 6)
    scope_changes = int(jira_metrics.get("scope_changes") or 1)
    sprint_velocity_points = float(jira_metrics.get("sprint_velocity") or max(8, commits * 0.8))

    normalized_velocity = max(0.5, min(1.5, sprint_velocity_points / max(contributors * 2.5, 1)))
    closure_rate = float(github_metrics.get("issue_closure_rate") or 0)
    creation_rate = float(github_metrics.get("issue_creation_rate") or max(bug_count, 1))
    completion_signal = max(20.0, min(95.0, 65 + (closure_rate - creation_rate) * 2 - blocked * 3))
    availability = max(60.0, min(100.0, 92 - blocked * 6 - open_prs * 1.5 - max(0, merge_hours - 24) * 0.2))

    return {
        "project_id": project_id,
        "project_name": project_name,
        "team_size": contributors,
        "planned_duration_days": 90,
        "sprint_velocity": round(normalized_velocity, 3),
        "tasks_completed_pct": round(completion_signal, 1),
        "scope_changes": scope_changes,
        "bugs_open": bug_count,
        "team_availability_pct": round(availability, 1),
        "days_elapsed": 42,
        "stakeholder_changes": int(jira_metrics.get("reopened_tickets") or 0),
        "commits_per_week": commits,
        "pull_requests_opened": open_prs,
        "pull_requests_merged": int(github_metrics.get("pull_requests_merged") or 0),
        "average_merge_time_hours": merge_hours,
        "blocked_issues": blocked,
        "bug_trend": bug_count,
        "code_churn": int(github_metrics.get("code_churn") or commits * 25),
    }


def score_normalized_project(project_record: Dict[str, object]) -> Dict[str, object]:
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    frame = engineer_features(pd.DataFrame([project_record]))
    score = float(model.predict_proba(frame[FEATURE_COLS])[:, 1][0])
    scored = frame.iloc[0].to_dict()
    scored["risk_score"] = score
    scored["risk_level"] = "Low" if score < 0.35 else "Medium" if score < 0.65 else "High"
    return scored


def demo_metrics() -> Dict[str, object]:
    return {
        "github": {
            "source": "github",
            "repository": "demo/deadline-warning",
            "commits_per_week": 18,
            "pull_requests_opened": 6,
            "open_pull_requests": 5,
            "pull_requests_merged": 4,
            "average_merge_time_hours": 32.5,
            "number_of_contributors": 5,
            "issue_creation_rate": 9,
            "issue_closure_rate": 4,
            "code_churn": 640,
        },
        "jira": {
            "source": "jira",
            "project_key": "DEMO",
            "story_points_completed": 42,
            "sprint_velocity": 21,
            "blocked_issues": 3,
            "reopened_tickets": 2,
            "bug_count": 14,
            "scope_changes": 4,
        },
    }


def run_ingestion(owner: Optional[str] = None, repo: Optional[str] = None) -> Dict[str, object]:
    github_metrics = None
    jira_metrics = None

    if owner and repo:
        github_metrics = GitHubConnector().fetch_repo_metrics(owner, repo)

    jira = JiraConnector()
    if jira.configured():
        jira_metrics = jira.fetch_project_metrics()

    if github_metrics is None and jira_metrics is None:
        metrics = demo_metrics()
        github_metrics = metrics["github"]
        jira_metrics = metrics["jira"]

    normalized = normalize_metrics(github_metrics, jira_metrics)
    scored = score_normalized_project(normalized)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": ["GitHub/Jira", "Feature Extraction", "Risk Prediction", "Dashboard"],
        "github": github_metrics,
        "jira": jira_metrics,
        "normalized_project": normalized,
        "scored_project": scored,
    }

    ENGINEERING_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    NORMALIZED_FEATURES_PATH.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    print(f"Engineering metrics saved -> {ENGINEERING_METRICS_PATH.relative_to(PROJECT_ROOT)}")
    return payload


if __name__ == "__main__":
    run_ingestion(os.getenv("GITHUB_OWNER"), os.getenv("GITHUB_REPO"))




