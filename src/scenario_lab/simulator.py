"""
simulator.py - Advanced Scenario Lab Engine.

Compares Current State with multiple named scenarios and returns dashboard-ready
comparison tables, risk deltas, radar chart data, and recommendation ranking.
"""

from typing import Any, Dict, List

from src.scenario_simulator import simulate


RADAR_FIELDS = ["team_size", "sprint_velocity", "scope_changes", "bugs_open", "planned_duration_days"]


class AdvancedScenarioLab:
    def run_multi_scenario(self, project_id: int, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        baseline = simulate(project_id, {})
        current_risk = float(baseline["current"]["risk_score"])
        current_features = baseline["current"]["features"]

        results = []
        comparison_table = [{
            "scenario": "Current State",
            "risk_score": round(current_risk * 100, 1),
            "risk_delta": 0.0,
            "improvement_pct": 0.0,
            "key_changes": "Baseline",
        }]

        for scenario in scenarios:
            name = scenario.get("name", "Unnamed Scenario")
            overrides = scenario.get("overrides", {})
            sim_result = simulate(project_id, overrides)
            scenario_risk = float(sim_result["scenario"]["risk_score"])
            risk_delta = scenario_risk - current_risk
            improvement_pct = max(0.0, (current_risk - scenario_risk) * 100)
            features = sim_result["scenario"]["features"]

            results.append({
                "name": name,
                "risk_score": scenario_risk,
                "risk_pct": round(scenario_risk * 100, 1),
                "risk_delta": risk_delta,
                "risk_delta_pct": round(risk_delta * 100, 1),
                "improvement_pct": round(improvement_pct, 1),
                "features": features,
                "recommendations": sim_result["scenario"].get("recommendations", []),
            })
            comparison_table.append({
                "scenario": name,
                "risk_score": round(scenario_risk * 100, 1),
                "risk_delta": round(risk_delta * 100, 1),
                "improvement_pct": round(improvement_pct, 1),
                "key_changes": ", ".join([f"{key}: {value}" for key, value in overrides.items()]) or "None",
            })

        ranked = sorted(results, key=lambda item: item["risk_score"])
        radar_chart = self._build_radar_chart(current_features, results)
        recommendation_ranking = [
            {
                "scenario": item["name"],
                "risk_pct": item["risk_pct"],
                "improvement_pct": item["improvement_pct"],
                "rank": rank + 1,
            }
            for rank, item in enumerate(ranked)
        ]

        return {
            "project_id": int(project_id),
            "current": {"risk_score": current_risk, "risk_pct": round(current_risk * 100, 1), "features": current_features},
            "scenarios": results,
            "comparison_table": comparison_table,
            "radar_chart": radar_chart,
            "recommendation_ranking": recommendation_ranking,
            "best_scenario": ranked[0]["name"] if ranked else "None",
        }

    @staticmethod
    def _build_radar_chart(current_features: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ["Team", "Velocity", "Scope Control", "Quality", "Timeline"]

        def normalize(features: Dict[str, Any]) -> List[float]:
            team = min(1.0, float(features.get("team_size", 0)) / 20)
            velocity = min(1.0, float(features.get("sprint_velocity", 0)) / 1.5)
            scope_control = 1.0 - min(1.0, float(features.get("scope_changes", 0)) / 8)
            quality = 1.0 - min(1.0, float(features.get("bugs_open", 0)) / 60)
            timeline = min(1.0, float(features.get("planned_duration_days", 0)) / 180)
            return [round(value * 100, 1) for value in [team, velocity, scope_control, quality, timeline]]

        datasets = [{"label": "Current State", "values": normalize(current_features)}]
        for item in scenarios:
            datasets.append({"label": item["name"], "values": normalize(item["features"])})
        return {"labels": labels, "datasets": datasets}
