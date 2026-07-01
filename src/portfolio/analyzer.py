"""
analyzer.py - Portfolio Risk Command Center.

Computes executive portfolio metrics, risk heatmaps, bottlenecks, and project
prioritization using business impact multiplied by risk probability.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"
DEPARTMENTS = ["Mobile", "Backend", "Web", "Data", "Platform", "Security"]
TEAMS = ["Alpha", "Beta", "Gamma", "Delta", "Omega"]


class PortfolioAnalyzer:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def analyze(self) -> Dict[str, Any]:
        df = pd.read_csv(PROJECTS_PATH)
        if df.empty or "risk_score" not in df.columns:
            return {"error": "Projects not scored. Run the training pipeline first."}

        df = df.copy()
        df["business_impact"] = self._deterministic_impact(df["project_id"])
        df["priority_score"] = df["business_impact"] * df["risk_score"]
        df["department"] = [DEPARTMENTS[(int(pid) - 1) % len(DEPARTMENTS)] for pid in df["project_id"]]
        df["team"] = [TEAMS[(int(pid) - 1) % len(TEAMS)] for pid in df["project_id"]]
        df["expected_delay_days"] = df.apply(self._estimate_delay_days, axis=1)
        df["forecast_trend"] = df.apply(self._trend_symbol, axis=1)
        df["resource_utilization"] = np.clip(
            df["tasks_completed_pct"] / df["team_availability_pct"].replace(0, np.nan), 0, 2
        ).fillna(0)

        portfolio_rows = df.nlargest(12, "priority_score").apply(self._command_row, axis=1).tolist()
        heatmaps = {
            "department_risk_heatmap": self._heatmap(df, "department"),
            "team_risk_heatmap": self._heatmap(df, "team"),
            "timeline_heatmap": self._timeline_heatmap(df),
        }

        bottlenecks = {
            "quality": df.nlargest(5, "bugs_open")[["project_id", "bugs_open", "risk_score"]].to_dict(orient="records"),
            "scope": df.nlargest(5, "scope_changes")[["project_id", "scope_changes", "risk_score"]].to_dict(orient="records"),
            "capacity": df.nsmallest(5, "team_availability_pct")[["project_id", "team_availability_pct", "risk_score"]].to_dict(orient="records"),
        }

        return {
            "portfolio_metrics": {
                "total_projects": int(len(df)),
                "portfolio_risk_score": round(float(df["risk_score"].mean()) * 100, 1),
                "high_risk_projects": int((df["risk_level"] == "High").sum()),
                "total_expected_delay_days": int(round(float(df["expected_delay_days"].sum()))),
                "resource_utilization": round(float(df["resource_utilization"].mean()) * 100, 1),
            },
            "command_center": portfolio_rows,
            "risk_distribution": df["risk_level"].value_counts().to_dict(),
            "heatmaps": heatmaps,
            "bottlenecks": bottlenecks,
            "top_priority_projects": df.nlargest(10, "priority_score")[
                ["project_id", "department", "team", "risk_level", "risk_score", "business_impact", "priority_score", "expected_delay_days"]
            ].to_dict(orient="records"),
            "prioritization_method": "business_impact x risk_probability",
        }

    @staticmethod
    def _deterministic_impact(project_ids: pd.Series) -> List[int]:
        return [int(((int(pid) * 37) % 10) + 1) for pid in project_ids]

    @staticmethod
    def _estimate_delay_days(row: pd.Series) -> int:
        completion_gap_delay = max(0.0, -float(row.get("completion_gap", 0)) * float(row.get("planned_duration_days", 90)))
        risk_delay = max(0.0, (float(row.get("risk_score", 0)) - 0.35) * 24)
        return int(round(completion_gap_delay + risk_delay))

    @staticmethod
    def _trend_symbol(row: pd.Series) -> str:
        if float(row.get("risk_score", 0)) >= 0.65 or float(row.get("completion_gap", 0)) < -0.1:
            return "up"
        if float(row.get("risk_score", 0)) < 0.35 and float(row.get("completion_gap", 0)) >= 0:
            return "down"
        return "flat"

    @staticmethod
    def _command_row(row: pd.Series) -> Dict[str, Any]:
        return {
            "project_id": int(row["project_id"]),
            "project": f"Project #{int(row['project_id'])}",
            "department": row["department"],
            "team": row["team"],
            "risk": round(float(row["risk_score"]) * 100, 1),
            "forecast": row["forecast_trend"],
            "delay_days": int(row["expected_delay_days"]),
            "business_impact": int(row["business_impact"]),
            "priority_score": round(float(row["priority_score"]), 3),
        }

    @staticmethod
    def _heatmap(df: pd.DataFrame, group_col: str) -> List[Dict[str, Any]]:
        grouped = df.groupby(group_col).agg(
            avg_risk=("risk_score", "mean"),
            high_risk=("risk_level", lambda values: int((values == "High").sum())),
            project_count=("project_id", "count"),
        ).reset_index()
        return [
            {
                group_col: row[group_col],
                "avg_risk": round(float(row["avg_risk"]) * 100, 1),
                "high_risk": int(row["high_risk"]),
                "project_count": int(row["project_count"]),
            }
            for _, row in grouped.iterrows()
        ]

    @staticmethod
    def _timeline_heatmap(df: pd.DataFrame) -> List[Dict[str, Any]]:
        bins = pd.cut(df["days_elapsed"], bins=[0, 30, 60, 90, 120, 180], labels=["0-30", "31-60", "61-90", "91-120", "121+"])
        grouped = df.assign(timeline_bucket=bins).groupby("timeline_bucket", observed=False)["risk_score"].mean().reset_index()
        return [
            {"period": str(row["timeline_bucket"]), "avg_risk": round(float(row["risk_score"]) * 100, 1)}
            for _, row in grouped.iterrows()
            if not pd.isna(row["risk_score"])
        ]
