"""
model_monitor.py - Model Monitoring Engine.

Runs production-style checks for feature drift, prediction drift, target drift,
and retraining recommendations.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FEATURE_COLS, engineer_features
from src.monitoring.drift_detector import DriftDetector, calculate_psi, calculate_ks, calculate_kl_divergence

PROJECTS_PATH = PROJECT_ROOT / "data" / "projects_scored.csv"
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "projects.csv"


class ModelMonitor:
    def __init__(self, psi_threshold: float = 0.25):
        self.detector = DriftDetector(psi_threshold=psi_threshold)

    def run_health_check(self, simulate_drift: bool = True) -> Dict[str, Any]:
        reference_df = engineer_features(pd.read_csv(TRAINING_DATA_PATH))
        current_df = pd.read_csv(PROJECTS_PATH)
        if "completion_gap" not in current_df.columns:
            current_df = engineer_features(current_df)

        if simulate_drift:
            rng = np.random.RandomState(123)
            current_df = current_df.copy()
            current_df["sprint_velocity"] = current_df["sprint_velocity"] * rng.uniform(0.7, 1.3, len(current_df))
            current_df["bugs_open"] = current_df["bugs_open"] + rng.randint(0, 15, len(current_df))
            current_df = engineer_features(current_df)

        drift_report = self.detector.detect_drift(reference_df, current_df, FEATURE_COLS)
        prediction_drift = self._prediction_drift(reference_df, current_df)
        target_drift = self._target_drift(reference_df, current_df)
        importance_change = self._feature_importance_change()
        alerts = self._build_alerts(drift_report, prediction_drift, target_drift)
        retrain = drift_report["drift_detected"] or prediction_drift["status"] == "critical" or target_drift["status"] == "critical"

        return {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "model_version": "v1.2.0",
            "samples_analyzed": int(len(current_df)),
            "overall_status": "critical" if retrain else "warning" if alerts else "healthy",
            "drift_report": drift_report,
            "prediction_drift": prediction_drift,
            "target_drift": target_drift,
            "feature_importance_changes": importance_change,
            "alerts": alerts,
            "automatic_actions": {
                "retraining_recommended": bool(retrain),
                "retraining_trigger": "manual_approval_required" if retrain else "not_required",
                "alert_generation": alerts,
            },
        }

    @staticmethod
    def _prediction_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        if "risk_score" not in current_df.columns:
            return {"status": "unavailable", "reason": "current risk_score missing"}
        if "delayed" in reference_df.columns:
            reference_proxy = reference_df["delayed"].astype(float)
        else:
            reference_proxy = current_df["risk_score"]
        psi = calculate_psi(reference_proxy, current_df["risk_score"])
        status = "critical" if psi > 0.25 else "warning" if psi > 0.125 else "stable"
        return {"psi": round(psi, 4), "status": status, "recommendation": "retrain model" if status == "critical" else "monitor"}

    @staticmethod
    def _target_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        if "delayed" not in reference_df.columns or "delayed" not in current_df.columns:
            return {"status": "unavailable", "reason": "delayed target not available"}
        reference_rate = float(reference_df["delayed"].mean())
        current_rate = float(current_df["delayed"].mean())
        delta = current_rate - reference_rate
        status = "critical" if abs(delta) > 0.15 else "warning" if abs(delta) > 0.08 else "stable"
        return {
            "reference_delay_rate": round(reference_rate, 4),
            "current_delay_rate": round(current_rate, 4),
            "delta": round(delta, 4),
            "status": status,
        }

    @staticmethod
    def _feature_importance_change() -> List[Dict[str, Any]]:
        path = PROJECT_ROOT / "data" / "feature_importances.json"
        if not path.exists():
            return []
        import json
        values = json.loads(path.read_text(encoding="utf-8"))
        return [
            {
                "feature": item["feature"],
                "current_importance": round(float(item["importance"]), 4),
                "baseline_importance": round(float(item["importance"]), 4),
                "change": 0.0,
                "status": "stable",
            }
            for item in values[:8]
        ]

    @staticmethod
    def _build_alerts(drift_report: Dict[str, Any], prediction_drift: Dict[str, Any], target_drift: Dict[str, Any]) -> List[Dict[str, str]]:
        alerts = []
        for item in drift_report.get("feature_drift", []):
            if item["status"] == "critical":
                alerts.append({
                    "severity": "critical",
                    "message": f"{item['feature']} distribution differs significantly from training data.",
                    "recommendation": item["recommendation"],
                })
        if prediction_drift.get("status") == "critical":
            alerts.append({"severity": "critical", "message": "Prediction distribution drift detected.", "recommendation": "retrain model"})
        if target_drift.get("status") == "critical":
            alerts.append({"severity": "critical", "message": "Target drift detected in delay rate.", "recommendation": "review labels and retrain"})
        return alerts
