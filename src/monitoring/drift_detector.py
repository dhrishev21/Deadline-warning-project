"""
drift_detector.py - Data Drift Monitoring.

Detects feature drift with PSI, KS statistic, and KL divergence. Thresholds are
configurable so cloud deployments can tune alert sensitivity.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if len(expected) == 0 or len(actual) == 0 or expected.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_percents = np.clip(expected_counts / max(expected_counts.sum(), 1), 0.0001, None)
    actual_percents = np.clip(actual_counts / max(actual_counts.sum(), 1), 0.0001, None)
    return float(np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents)))


def calculate_ks(expected: pd.Series, actual: pd.Series) -> float:
    expected = np.sort(expected.dropna().astype(float).to_numpy())
    actual = np.sort(actual.dropna().astype(float).to_numpy())
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([expected, actual])))
    expected_cdf = np.searchsorted(expected, values, side="right") / len(expected)
    actual_cdf = np.searchsorted(actual, values, side="right") / len(actual)
    return float(np.max(np.abs(expected_cdf - actual_cdf)))


def calculate_kl_divergence(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    min_value = min(float(expected.min()), float(actual.min()))
    max_value = max(float(expected.max()), float(actual.max()))
    if min_value == max_value:
        return 0.0
    bins = np.linspace(min_value, max_value, buckets + 1)
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    p = np.clip(expected_counts / max(expected_counts.sum(), 1), 0.0001, None)
    q = np.clip(actual_counts / max(actual_counts.sum(), 1), 0.0001, None)
    return float(np.sum(p * np.log(p / q)))


class DriftDetector:
    def __init__(self, psi_threshold: float = 0.25, ks_threshold: float = 0.2, kl_threshold: float = 0.15):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.kl_threshold = kl_threshold

    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        drift_results = []
        critical_features = []

        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            ref_series = reference_data[feature].dropna()
            cur_series = current_data[feature].dropna()
            if len(ref_series) == 0 or len(cur_series) == 0:
                continue

            psi = calculate_psi(ref_series, cur_series)
            ks = calculate_ks(ref_series, cur_series)
            kl = calculate_kl_divergence(ref_series, cur_series)
            critical = psi > self.psi_threshold or ks > self.ks_threshold or kl > self.kl_threshold
            warning = psi > self.psi_threshold / 2 or ks > self.ks_threshold / 2 or kl > self.kl_threshold / 2
            status = "critical" if critical else "warning" if warning else "stable"
            if critical:
                critical_features.append(feature)

            drift_results.append({
                "feature": feature,
                "psi": round(psi, 4),
                "ks_statistic": round(ks, 4),
                "kl_divergence": round(kl, 4),
                "status": status,
                "recommendation": "retrain model" if critical else "monitor" if warning else "no action",
            })

        return {
            "drift_detected": bool(critical_features),
            "overall_status": "critical" if critical_features else "warning" if any(item["status"] == "warning" for item in drift_results) else "stable",
            "critical_features": critical_features,
            "feature_drift": drift_results,
            "thresholds": {
                "psi": self.psi_threshold,
                "ks_statistic": self.ks_threshold,
                "kl_divergence": self.kl_threshold,
            },
        }
