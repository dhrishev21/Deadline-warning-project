"""
generate_advanced_artifacts.py
Generates example JSON responses and static charts for advanced platform features.
Run: python src/generate_advanced_artifacts.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.project_assistant import ProjectAssistant
from src.forecasting.xgboost_model import XGBoostRiskForecaster
from src.monte_carlo.simulator import MonteCarloSimulator
from src.portfolio.analyzer import PortfolioAnalyzer
from src.scenario_lab.simulator import AdvancedScenarioLab
from src.similarity.historical_similarity import find_similar_projects
from src.monitoring.model_monitor import ModelMonitor

DATA_PATH = PROJECT_ROOT / "data" / "advanced_feature_examples.json"
FORECAST_CHART = PROJECT_ROOT / "charts" / "risk_forecast.png"
MONTE_CARLO_CHART = PROJECT_ROOT / "charts" / "monte_carlo_delivery.png"


def generate(project_id: int = 1):
    forecast = XGBoostRiskForecaster().forecast(project_id, 4)
    monte_carlo = MonteCarloSimulator(seed=7).run(project_id, 1000)
    similar = find_similar_projects(project_id, 5)
    scenario_lab = AdvancedScenarioLab().run_multi_scenario(project_id, [
        {"name": "Scenario A: Add 2 Developers", "overrides": {"team_size": 12}},
        {"name": "Scenario B: Reduce Scope", "overrides": {"scope_changes": 1}},
        {"name": "Scenario C: Extend Deadline", "overrides": {"deadline_extension_days": 14}},
    ])
    portfolio = PortfolioAnalyzer().analyze()
    drift = ModelMonitor().run_health_check(simulate_drift=True)
    assistant = ProjectAssistant(provider="local").ask(project_id, "Why is this project risky?")

    payload = {
        "forecast": forecast,
        "monte_carlo": monte_carlo,
        "similar_projects": similar,
        "scenario_lab": scenario_lab,
        "portfolio": portfolio,
        "drift": drift,
        "assistant": assistant,
    }
    DATA_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    save_forecast_chart(forecast)
    save_monte_carlo_chart(monte_carlo)
    print(f"Saved {DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved {FORECAST_CHART.relative_to(PROJECT_ROOT)}")
    print(f"Saved {MONTE_CARLO_CHART.relative_to(PROJECT_ROOT)}")


def save_forecast_chart(forecast):
    weeks = [0] + [item["week"] for item in forecast["forecast"]]
    risks = [forecast["current_risk"]] + [item["risk"] for item in forecast["forecast"]]
    lower = [forecast["current_risk"]] + [item["confidence_lower"] for item in forecast["forecast"]]
    upper = [forecast["current_risk"]] + [item["confidence_upper"] for item in forecast["forecast"]]
    plt.figure(figsize=(8, 4.5))
    plt.fill_between(weeks, lower, upper, color="#38bdf8", alpha=0.18, label="Confidence band")
    plt.plot(weeks, risks, color="#f43f5e", linewidth=2.5, marker="o", label="Forecast risk")
    plt.axhline(65, color="#f43f5e", linestyle="--", alpha=0.6, label="High risk threshold")
    plt.title("Risk Timeline Forecast", fontweight="bold")
    plt.xlabel("Weeks Ahead")
    plt.ylabel("Risk (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FORECAST_CHART, dpi=150)
    plt.close()


def save_monte_carlo_chart(monte_carlo):
    cdf = monte_carlo["cdf"]
    labels = [item["day"] for item in cdf]
    probs = [item["probability"] * 100 for item in cdf]
    plt.figure(figsize=(8, 4.5))
    plt.plot(labels, probs, color="#10b981", linewidth=2.5)
    for marker in monte_carlo["percentile_markers"]:
        plt.axvline(marker["days_from_start"], linestyle="--", alpha=0.45)
        plt.text(marker["days_from_start"], 5, f"P{marker['percentile']}", rotation=90, va="bottom")
    plt.title("Monte Carlo Delivery Probability", fontweight="bold")
    plt.xlabel("Completion Day From Project Start")
    plt.ylabel("Cumulative Probability (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(MONTE_CARLO_CHART, dpi=150)
    plt.close()


if __name__ == "__main__":
    generate()
