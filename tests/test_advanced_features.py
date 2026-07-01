import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.forecasting.xgboost_model import XGBoostRiskForecaster
from src.monte_carlo.simulator import MonteCarloSimulator
from src.similarity.historical_similarity import find_similar_projects
from src.scenario_lab.simulator import AdvancedScenarioLab
from src.portfolio.analyzer import PortfolioAnalyzer
from src.monitoring.model_monitor import ModelMonitor
from src.assistant.project_assistant import ProjectAssistant
from src.serve_dashboard import app


class AdvancedPlatformTests(unittest.TestCase):
    def test_forecast_contract(self):
        payload = XGBoostRiskForecaster().forecast(1, 4)
        self.assertEqual(payload["project_id"], 1)
        self.assertEqual(len(payload["forecast"]), 4)
        self.assertIn("confidence_lower", payload["forecast"][0])
        self.assertIn("visualization", payload)

    def test_monte_carlo_contract(self):
        payload = MonteCarloSimulator(seed=7).run(1, 1000)
        self.assertEqual(payload["simulations"], 1000)
        self.assertIn("p90_completion", payload)
        self.assertIn("histogram", payload)
        self.assertGreater(len(payload["cdf"]), 0)

    def test_similarity_contract(self):
        payload = find_similar_projects(1, top_n=3)
        self.assertEqual(len(payload["similar_projects"]), 3)
        self.assertIn("insight", payload)

    def test_scenario_lab_contract(self):
        payload = AdvancedScenarioLab().run_multi_scenario(1, [
            {"name": "Add capacity", "overrides": {"team_size": 12}},
            {"name": "Reduce scope", "overrides": {"scope_changes": 1}},
        ])
        self.assertEqual(len(payload["scenarios"]), 2)
        self.assertIn("radar_chart", payload)
        self.assertIn("best_scenario", payload)

    def test_portfolio_contract(self):
        payload = PortfolioAnalyzer().analyze()
        self.assertEqual(payload["portfolio_metrics"]["total_projects"], 300)
        self.assertIn("command_center", payload)
        self.assertIn("heatmaps", payload)

    def test_monitoring_contract(self):
        payload = ModelMonitor().run_health_check(simulate_drift=False)
        self.assertIn("drift_report", payload)
        self.assertIn("prediction_drift", payload)
        self.assertIn("automatic_actions", payload)

    def test_assistant_fallback_contract(self):
        payload = ProjectAssistant(provider="local").ask(1, "Why is this project risky?")
        self.assertIn("response", payload)
        self.assertGreater(len(payload["citations"]), 0)

    def test_api_contracts(self):
        client = TestClient(app)
        self.assertEqual(client.get("/api/forecast", params={"project_id": 1}).status_code, 200)
        self.assertEqual(client.get("/api/monte-carlo", params={"project_id": 1, "simulations": 1000}).status_code, 200)
        self.assertEqual(client.get("/api/similar-projects", params={"project_id": 1}).status_code, 200)
        self.assertEqual(client.get("/api/portfolio").status_code, 200)
        self.assertEqual(client.get("/api/drift").status_code, 200)
        scenario = {"project_id": 1, "scenarios": [{"name": "A", "overrides": {"team_size": 12}}]}
        self.assertEqual(client.post("/api/scenario-lab", json=scenario).status_code, 200)
        question = {"project_id": 1, "question": "Why is this project risky?"}
        self.assertEqual(client.post("/api/assistant", json=question).status_code, 200)


if __name__ == "__main__":
    unittest.main()
