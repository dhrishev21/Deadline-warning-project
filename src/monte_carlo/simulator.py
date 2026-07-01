"""
simulator.py - Monte Carlo Deadline Simulation Engine.

Randomizes delivery variables to estimate completion-date distributions and
probability of meeting a deadline.
"""

import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "delay_model.pkl"
PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"


def _load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def _load_project(project_id: int) -> pd.Series:
    df = pd.read_csv(PROJECTS_PATH)
    match = df[df["project_id"] == int(project_id)]
    if match.empty:
        raise ValueError(f"Project {project_id} not found.")
    return match.iloc[0]


class MonteCarloSimulator:
    """Estimate delivery likelihood from randomized project execution paths."""

    def __init__(self, seed: int = 42):
        self.model = _load_model()
        self.rng = np.random.RandomState(seed)

    def run(self, project_id: int, n_simulations: int = 5000, deadline_date: Optional[str] = None) -> Dict[str, object]:
        allowed = {1000, 5000, 10000}
        if n_simulations not in allowed:
            n_simulations = min(allowed, key=lambda value: abs(value - int(n_simulations)))

        row = _load_project(project_id)
        planned_days = int(row["planned_duration_days"])
        days_elapsed = int(row["days_elapsed"])
        remaining_days = max(1, planned_days - days_elapsed)
        tasks_done_pct = float(row["tasks_completed_pct"])
        remaining_work = max(5.0, 100.0 - tasks_done_pct)

        today = datetime.now(timezone.utc).date()
        project_start = today - timedelta(days=days_elapsed)
        deadline = datetime.fromisoformat(deadline_date).date() if deadline_date else today + timedelta(days=remaining_days)
        deadline_days_from_start = (deadline - project_start).days

        completion_days = []
        for _ in range(n_simulations):
            productivity = float(self.rng.lognormal(mean=0.0, sigma=0.25))
            rework_pct = float(self.rng.beta(2, 8))
            availability = float(self.rng.normal(loc=float(row["team_availability_pct"]) / 100, scale=0.08))
            availability = max(0.4, min(1.0, availability))
            weekly_bugs = int(self.rng.poisson(lam=max(1, float(row["bugs_open"]) * 0.15)))
            scope_creep_rate = min(0.35, 0.08 + float(row["scope_changes"]) * 0.018)

            work_remaining = remaining_work
            sim_days = 0
            max_sim_days = remaining_days * 4
            weekly_capacity = (
                float(row["sprint_velocity"])
                * productivity
                * availability
                * (1.0 - rework_pct)
                * float(row["team_size"])
                * 1.2
            )
            weekly_capacity = max(1.0, weekly_capacity)

            while work_remaining > 0 and sim_days < max_sim_days:
                bug_overhead = weekly_bugs * 0.5
                effective_progress = max(0.5, weekly_capacity - bug_overhead)
                work_remaining -= effective_progress
                sim_days += 7

                if self.rng.random() < scope_creep_rate:
                    work_remaining += self.rng.uniform(2, 8)
                weekly_bugs = int(self.rng.poisson(lam=max(1, weekly_bugs * 0.9)))

            completion_days.append(days_elapsed + sim_days)

        completion_arr = np.array(completion_days, dtype=float)
        p10 = float(np.percentile(completion_arr, 10))
        p50 = float(np.percentile(completion_arr, 50))
        p80 = float(np.percentile(completion_arr, 80))
        p90 = float(np.percentile(completion_arr, 90))
        expected_delay = max(0.0, float(np.mean(completion_arr)) - planned_days)
        delivery_probability = float(np.mean(completion_arr <= deadline_days_from_start))

        hist_counts, hist_edges = np.histogram(completion_arr, bins=25)
        histogram = []
        for idx in range(len(hist_counts)):
            start_day = float(hist_edges[idx])
            end_day = float(hist_edges[idx + 1])
            histogram.append({
                "bin_start": round(start_day, 1),
                "bin_end": round(end_day, 1),
                "date_start": (project_start + timedelta(days=start_day)).isoformat(),
                "date_end": (project_start + timedelta(days=end_day)).isoformat(),
                "count": int(hist_counts[idx]),
            })

        sorted_days = np.sort(completion_arr)
        cdf_step = max(1, len(sorted_days) // 30)
        cdf_data = []
        for idx in range(0, len(sorted_days), cdf_step):
            day_value = float(sorted_days[idx])
            cdf_data.append({
                "day": int(day_value),
                "date": (project_start + timedelta(days=day_value)).isoformat(),
                "probability": round(float((idx + 1) / n_simulations), 4),
            })

        return {
            "project_id": int(project_id),
            "simulations": int(n_simulations),
            "planned_duration_days": planned_days,
            "deadline_date": deadline.isoformat(),
            "expected_delay_days": round(expected_delay, 1),
            "delivery_probability": round(delivery_probability, 4),
            "probability_on_time": round(delivery_probability, 4),
            "p50_days": round(p50, 1),
            "p80_days": round(p80, 1),
            "p90_days": round(p90, 1),
            "p50_completion": (project_start + timedelta(days=p50)).isoformat(),
            "p80_completion": (project_start + timedelta(days=p80)).isoformat(),
            "p90_completion": (project_start + timedelta(days=p90)).isoformat(),
            "confidence_interval": {
                "p10_completion": (project_start + timedelta(days=p10)).isoformat(),
                "p90_completion": (project_start + timedelta(days=p90)).isoformat(),
            },
            "percentile_markers": [
                {"percentile": 50, "date": (project_start + timedelta(days=p50)).isoformat(), "days_from_start": round(p50, 1)},
                {"percentile": 80, "date": (project_start + timedelta(days=p80)).isoformat(), "days_from_start": round(p80, 1)},
                {"percentile": 90, "date": (project_start + timedelta(days=p90)).isoformat(), "days_from_start": round(p90, 1)},
            ],
            "histogram": histogram,
            "cdf": cdf_data,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
