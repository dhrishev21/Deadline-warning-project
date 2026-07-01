"""
prompt_templates.py — Prompt templates for the project assistant.

Contains structured prompts for different question types.
"""

SYSTEM_PROMPT = """You are an AI Project Risk Analyst. You answer questions about project health,
risk factors, and delivery predictions using ONLY the provided context data.

Rules:
- Only use facts from the provided context
- Cite specific metrics and data sources
- If you don't have enough data, say so
- Be concise and actionable
- Format responses with bullet points for clarity"""

RISK_ANALYSIS_TEMPLATE = """Based on the following project data, explain why this project is at {risk_level} risk:

Project: #{project_id}
Risk Score: {risk_score}%
Team Size: {team_size}
Sprint Velocity: {sprint_velocity}
Bugs Open: {bugs_open}
Scope Changes: {scope_changes}
Tasks Completed: {tasks_completed_pct}%
Team Availability: {team_availability_pct}%
Completion Gap: {completion_gap}

Top Risk Drivers:
{risk_drivers}

Similar Projects Insight:
{similarity_insight}

Provide a clear, data-driven explanation."""

RECOMMENDATION_TEMPLATE = """Given the current project state and these recommendations:

{recommendations}

Summarize the most impactful actions the team should take, ranked by effectiveness."""

FORECAST_TEMPLATE = """The risk forecast for Project #{project_id} shows:

Current Risk: {current_risk}%
Trend: {risk_trend}
{forecast_details}

Explain what this forecast means for the project timeline."""

WHAT_IF_TEMPLATE = """The user asks: "{question}"

Current project state for Project #{project_id}:
{current_state}

Use the scenario simulation capabilities to answer this question.
If the question involves changing team size, scope, velocity, or deadlines,
explain the expected impact on risk."""
