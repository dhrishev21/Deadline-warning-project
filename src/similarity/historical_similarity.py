"""
historical_similarity.py - Historical Project Matching Engine.

Finds projects with comparable delivery conditions using cosine similarity,
nearest-neighbor distance, or an embedding-style weighted feature vector.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import engineer_features

PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"

SIMILARITY_FEATURES = [
    "team_size",
    "sprint_velocity",
    "scope_changes",
    "bugs_open",
    "tasks_completed_pct",
    "team_availability_pct",
    "planned_duration_days",
    "bug_density",
    "completion_gap",
    "scope_risk",
    "days_elapsed",
]

PROJECT_NAMES = [
    "Phoenix", "Atlas", "Neptune", "Horizon", "Quantum", "Apollo",
    "Titan", "Orion", "Vanguard", "Eclipse", "Mercury", "Zenith",
    "Falcon", "Delta", "Omega", "Nova", "Vertex", "Summit",
    "Pioneer", "Cascade", "Spectra", "Nexus", "Prism", "Helix",
]

TECH_STACKS = ["Python/API", "React/Web", "Mobile", "Data Platform", "ERP", "Cloud Migration"]


def _get_project_name(project_id: int) -> str:
    idx = (int(project_id) - 1) % len(PROJECT_NAMES)
    return f"Project {PROJECT_NAMES[idx]}"


def _synthetic_context(row: pd.Series) -> Dict[str, object]:
    complexity = min(10, max(1, int(row.get("scope_changes", 0)) + int(row.get("bugs_open", 0)) // 8 + 2))
    deadline_pressure = max(0.0, min(1.0, float(row.get("days_elapsed", 0)) / max(float(row.get("planned_duration_days", 1)), 1)))
    issue_trend = "rising" if float(row.get("bugs_open", 0)) > 25 or float(row.get("scope_changes", 0)) > 4 else "stable"
    return {
        "architecture_complexity": complexity,
        "technology_stack": TECH_STACKS[(int(row["project_id"]) - 1) % len(TECH_STACKS)],
        "deadline_pressure": round(deadline_pressure, 3),
        "issue_trend": issue_trend,
    }


def _build_similarity_scores(feature_matrix: np.ndarray, target_idx: int, algorithm: str) -> np.ndarray:
    algorithm = algorithm.lower()
    if algorithm == "nearest_neighbors":
        nn = NearestNeighbors(n_neighbors=len(feature_matrix), metric="euclidean")
        nn.fit(feature_matrix)
        distances, indices = nn.kneighbors(feature_matrix[target_idx].reshape(1, -1))
        scores = np.zeros(len(feature_matrix), dtype=float)
        max_distance = max(float(distances[0].max()), 1e-9)
        for distance, idx in zip(distances[0], indices[0]):
            scores[idx] = 1.0 - (float(distance) / max_distance)
        return scores

    if algorithm == "embedding_similarity":
        weights = np.array([1.1, 1.4, 1.3, 1.2, 1.0, 0.9, 0.8, 1.1, 1.5, 1.2, 0.7])
        weighted = feature_matrix * weights[: feature_matrix.shape[1]]
        return cosine_similarity(weighted[target_idx].reshape(1, -1), weighted)[0]

    return cosine_similarity(feature_matrix[target_idx].reshape(1, -1), feature_matrix)[0]


def find_similar_projects(
    project_id: int,
    top_n: int = 5,
    min_similarity: float = 0.0,
    algorithm: str = "cosine_similarity",
) -> Dict[str, object]:
    df = pd.read_csv(PROJECTS_PATH)
    missing = [col for col in SIMILARITY_FEATURES if col not in df.columns]
    if missing:
        df = engineer_features(df)

    target_mask = df["project_id"] == int(project_id)
    if not target_mask.any():
        raise ValueError(f"Project {project_id} not found.")

    available_features = [feature for feature in SIMILARITY_FEATURES if feature in df.columns]
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(df[available_features].fillna(0))
    target_idx = int(df[target_mask].index[0])
    similarities = _build_similarity_scores(feature_matrix, target_idx, algorithm)

    results: List[Dict[str, object]] = []
    for idx in np.argsort(similarities)[::-1]:
        if idx == target_idx:
            continue
        sim_score = max(0.0, min(1.0, float(similarities[idx])))
        if sim_score < min_similarity:
            continue

        row = df.iloc[idx]
        delayed = bool(row.get("delayed", 0))
        completion_gap = float(row.get("completion_gap", 0))
        estimated_delay = max(0, int(-completion_gap * float(row.get("planned_duration_days", 90))))
        context = _synthetic_context(row)
        results.append({
            "project_id": int(row["project_id"]),
            "project_name": _get_project_name(int(row["project_id"])),
            "similarity_pct": round(sim_score * 100, 1),
            "risk_score": round(float(row.get("risk_score", 0)) * 100, 1),
            "risk_level": str(row.get("risk_level", "Unknown")),
            "delayed": delayed,
            "outcome": f"Finished {estimated_delay} days late" if delayed else "Finished on time",
            "estimated_delay_days": estimated_delay,
            "team_size": int(row["team_size"]),
            "sprint_velocity": round(float(row["sprint_velocity"]), 2),
            **context,
        })
        if len(results) >= top_n:
            break

    high_sim = [item for item in results if item["similarity_pct"] >= 85]
    if high_sim:
        late_count = sum(1 for item in high_sim if item["delayed"])
        late_pct = round(late_count / len(high_sim) * 100, 0)
        avg_delay = round(float(np.mean([item["estimated_delay_days"] for item in high_sim])), 1)
        insight = f"Projects with >85% similarity finished late {late_pct:.0f}% of the time (avg delay: {avg_delay} days)."
    else:
        late_count = sum(1 for item in results if item["delayed"])
        late_pct = round(late_count / max(len(results), 1) * 100, 0)
        insight = f"Nearest historical matches finished late {late_pct:.0f}% of the time."

    target_row = df.iloc[target_idx]
    return {
        "project_id": int(project_id),
        "project_name": _get_project_name(int(project_id)),
        "current_risk": round(float(target_row.get("risk_score", 0)) * 100, 1),
        "similar_projects": results,
        "insight": insight,
        "features_used": available_features,
        "algorithm": algorithm,
    }
