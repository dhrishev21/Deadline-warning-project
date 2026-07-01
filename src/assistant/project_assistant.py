"""
project_assistant.py - Natural Language Project Assistant.

Pipeline: question -> context retrieval -> constrained reasoning -> structured
response. OpenAI/LangChain are optional; deterministic local fallback keeps demos
working without credentials.
"""

import os
from typing import Any, Dict, List

from src.assistant.retrieval import retrieve_portfolio_context, retrieve_project_context

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import langchain  # noqa: F401
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False


class ProjectAssistant:
    def __init__(self, provider: str = "auto"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.provider = provider
        self.use_openai = provider in {"auto", "openai"} and OPENAI_AVAILABLE and bool(self.api_key)

    def ask(self, project_id: int, question: str) -> Dict[str, Any]:
        context = retrieve_project_context(project_id)
        if self._is_portfolio_question(question):
            context["portfolio_context"] = retrieve_portfolio_context()
            context["sources"].extend(context["portfolio_context"].get("sources", []))

        if self.use_openai:
            response_text = self._ask_openai(question, context)
            generation_mode = "openai"
        else:
            response_text = self._rule_based_fallback(question, context)
            generation_mode = "local_rule_based"

        return {
            "project_id": int(project_id),
            "question": question,
            "response": response_text,
            "citations": self._citations(context),
            "sources_used": sorted(set(context.get("sources", []))),
            "generation_mode": generation_mode,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "guardrails": ["retrieved_context_only", "no_external_facts", "source_citations_required"],
        }

    @staticmethod
    def _is_portfolio_question(question: str) -> bool:
        q = question.lower()
        return any(term in q for term in ["which project", "portfolio", "immediate attention", "highest risk", "all projects"])

    def _ask_openai(self, question: str, context: Dict[str, Any]) -> str:
        try:
            client = OpenAI(api_key=self.api_key)
            system_prompt = (
                "You are an AI Project Risk Analyst. Answer using ONLY the provided JSON context. "
                "Cite source names in brackets. If context is insufficient, say so. Keep under 160 words."
            )
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.1,
                max_tokens=320,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self._rule_based_fallback(question, context)

    def _rule_based_fallback(self, question: str, context: Dict[str, Any]) -> str:
        q = question.lower()
        metrics = context.get("metrics", {})
        lines: List[str] = []

        if "immediate attention" in q or "which project" in q:
            top = context.get("portfolio_context", {}).get("portfolio", {}).get("top_risk_projects", [])
            if top:
                project = top[0]
                return f"Project #{project['project_id']} needs immediate attention because it has the highest retrieved risk score ({project['risk_score']:.1%}) [portfolio_metrics]."

        if "what happens" in q or "add" in q or "developers" in q:
            recs = context.get("recommendations", [])
            if recs:
                top = recs[0]
                return (
                    f"The closest retrieved action is: {top['recommendation']}. "
                    f"It is expected to reduce risk by {top.get('expected_risk_reduction', 0) * 100:.1f} percentage points "
                    f"[recommendation_engine]."
                )

        if "late" in q or "days" in q:
            similar = context.get("similar_projects", {})
            matches = similar.get("matches", [])
            if matches:
                avg_delay = sum(item.get("estimated_delay_days", 0) for item in matches) / max(len(matches), 1)
                return f"Similar historical projects averaged about {avg_delay:.1f} delayed days [historical_similarity]."

        if "forecast" in q or "future" in q or "evolve" in q:
            forecast = context.get("forecast", {})
            weeks = forecast.get("weeks", [])
            if weeks:
                trend = forecast.get("trend", "unknown")
                week_text = ", ".join([f"week {item['week']}: {item['risk']}%" for item in weeks])
                escalation = forecast.get("escalation") or "No high-risk crossing predicted in the horizon."
                return f"Risk is {trend}. Forecast: {week_text}. {escalation} [risk_forecast]."

        if "why" in q or "risky" in q or "risk" in q:
            drivers = context.get("risk_drivers", {}).get("positive", [])
            lines.append(f"Project #{context['project_id']} is {metrics.get('risk_level', 'Unknown')} risk with score {metrics.get('risk_score', 0) * 100:.1f}% [project_metrics].")
            if drivers:
                lines.append("Primary drivers [shap_explanations]:")
                lines.extend([f"- {driver}" for driver in drivers])
            return "\n".join(lines)

        return (
            f"Project #{context['project_id']} is {metrics.get('risk_level', 'Unknown')} risk "
            f"({metrics.get('risk_score', 0) * 100:.1f}%). Sprint velocity is {metrics.get('sprint_velocity', 0)} "
            f"and open bugs are {metrics.get('bugs_open', 0)} [project_metrics]."
        )

    @staticmethod
    def _citations(context: Dict[str, Any]) -> List[Dict[str, str]]:
        labels = {
            "project_metrics": "data/projects_scored.csv",
            "shap_explanations": "data/shap_values.json",
            "recommendation_engine": "data/recommendations.json",
            "risk_forecast": "src/forecasting/forecaster.py",
            "historical_similarity": "src/similarity/historical_similarity.py",
            "portfolio_metrics": "src/portfolio/analyzer.py",
        }
        return [{"source": source, "path": labels.get(source, source)} for source in sorted(set(context.get("sources", [])))]
