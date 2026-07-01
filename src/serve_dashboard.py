"""
serve_dashboard.py
FastAPI server for the AI Project Risk Intelligence Platform.
Run: python src/serve_dashboard.py
"""

import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

from src.assistant.project_assistant import ProjectAssistant
from src.feature_pipeline import run_ingestion
from src.forecasting.lstm_model import LSTMForecaster
from src.forecasting.prophet_model import ProphetForecaster
from src.forecasting.xgboost_model import XGBoostRiskForecaster
from src.monitoring.model_monitor import ModelMonitor
from src.monte_carlo.simulator import MonteCarloSimulator
from src.portfolio.analyzer import PortfolioAnalyzer
from src.scenario_lab.simulator import AdvancedScenarioLab
from src.scenario_simulator import simulate
from src.similarity.historical_similarity import find_similar_projects

app = FastAPI(
    title="AI Project Risk Intelligence API",
    version="2.0.0",
    description="Risk prediction, explainability, forecasting, simulation, portfolio analytics, monitoring, and assistant APIs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioRequest(BaseModel):
    project_id: int
    overrides: Dict[str, Any]


class AssistantRequest(BaseModel):
    project_id: int
    question: str


class ScenarioLabRequest(BaseModel):
    project_id: int
    scenarios: List[Dict[str, Any]]


def _read_json(relative_path: str) -> Any:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing {relative_path}. Run the data pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/projects")
async def get_projects():
    try:
        return {
            "projects": _read_json("data/projects_scored.json"),
            "shap": _read_json("data/shap_values.json"),
            "recommendations": _read_json("data/recommendations.json"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/explanations")
async def get_explanations(project_id: Optional[int] = None):
    try:
        payload = _read_json("data/shap_values.json")
        if project_id is None:
            return payload
        selected = [item for item in payload.get("projects", []) if int(item["project_id"]) == int(project_id)]
        if not selected:
            raise HTTPException(status_code=404, detail=f"No explanation found for project {project_id}")
        return selected[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/recommendations")
async def get_recommendations(project_id: Optional[int] = None):
    try:
        payload = _read_json("data/recommendations.json")
        if project_id is None:
            return payload
        selected = [item for item in payload.get("projects", []) if int(item["project_id"]) == int(project_id)]
        if not selected:
            raise HTTPException(status_code=404, detail=f"No recommendations found for project {project_id}")
        return selected[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/scenario")
async def simulate_scenario(req: ScenarioRequest):
    try:
        return simulate(req.project_id, req.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/sync")
async def sync_data(owner: Optional[str] = None, repo: Optional[str] = None):
    try:
        return run_ingestion(owner or os.getenv("GITHUB_OWNER"), repo or os.getenv("GITHUB_REPO"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/engineering-metrics")
async def get_engineering_metrics():
    metrics_file = DATA_DIR / "engineering_metrics.json"
    if metrics_file.exists():
        return _read_json("data/engineering_metrics.json")
    return run_ingestion()


@app.get("/api/forecast")
async def get_forecast(project_id: int, horizon_weeks: int = 4, model: str = "xgboost"):
    try:
        model_key = model.lower()
        if model_key == "prophet":
            forecaster = ProphetForecaster()
        elif model_key == "lstm":
            forecaster = LSTMForecaster()
        else:
            forecaster = XGBoostRiskForecaster()
        return forecaster.forecast(project_id, horizon_weeks)
    except ImportError:
        return XGBoostRiskForecaster().forecast(project_id, horizon_weeks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/monte-carlo")
async def get_monte_carlo(project_id: int, simulations: int = 5000, deadline_date: Optional[str] = None):
    try:
        return MonteCarloSimulator().run(project_id, simulations, deadline_date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/similar-projects")
async def get_similar_projects(project_id: int, top_n: int = 5, algorithm: str = "cosine_similarity"):
    try:
        return find_similar_projects(project_id, top_n, algorithm=algorithm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/assistant")
async def ask_assistant(req: AssistantRequest):
    try:
        return ProjectAssistant().ask(req.project_id, req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/scenario-lab")
async def run_scenario_lab(req: ScenarioLabRequest):
    try:
        return AdvancedScenarioLab().run_multi_scenario(req.project_id, req.scenarios)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/portfolio")
async def get_portfolio():
    try:
        return PortfolioAnalyzer().analyze()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/drift")
async def check_drift(simulate_drift: bool = True):
    try:
        return ModelMonitor().run_health_check(simulate_drift=simulate_drift)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
async def serve_index():
    return FileResponse(PROJECT_ROOT / "index.html")


app.mount("/", StaticFiles(directory=PROJECT_ROOT), name="static")


def open_browser():
    time.sleep(1.5)
    print("\n>>> Opening dashboard at http://localhost:8000/ ...")
    webbrowser.open("http://localhost:8000/")


def main():
    os.chdir(PROJECT_ROOT)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("src.serve_dashboard:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
