"""
jira_connector.py
Jira integration layer for sprint and issue health metrics.
Credentials:
  JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import requests
from requests.auth import HTTPBasicAuth


class JiraConnector:
    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        project_key: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("JIRA_BASE_URL", "")).rstrip("/")
        self.email = email or os.getenv("JIRA_EMAIL")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN")
        self.project_key = project_key or os.getenv("JIRA_PROJECT_KEY")

    def configured(self) -> bool:
        return bool(self.base_url and self.email and self.api_token and self.project_key)

    def _search(self, jql: str, fields: str = "summary,status,issuetype,customfield_10016,updated") -> Dict[str, object]:
        if not self.configured():
            raise ValueError("Jira connector is not configured. Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, and JIRA_PROJECT_KEY.")

        response = requests.get(
            f"{self.base_url}/rest/api/3/search",
            auth=HTTPBasicAuth(self.email, self.api_token),
            headers={"Accept": "application/json"},
            params={"jql": jql, "fields": fields, "maxResults": 100},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def fetch_project_metrics(self, lookback_days: int = 14) -> Dict[str, object]:
        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        project_jql = f'project = "{self.project_key}" AND updated >= "{since}"'
        issues = self._search(project_jql).get("issues", [])

        story_points = 0.0
        blocked_issues = 0
        reopened_tickets = 0
        bug_count = 0
        scope_changes = 0

        for issue in issues:
            fields = issue.get("fields", {})
            status_name = fields.get("status", {}).get("name", "").lower()
            issue_type = fields.get("issuetype", {}).get("name", "").lower()
            summary = fields.get("summary", "").lower()

            if status_name in {"done", "closed", "resolved"}:
                story_points += float(fields.get("customfield_10016") or 0)
            if "blocked" in status_name or "blocked" in summary:
                blocked_issues += 1
            if "reopen" in status_name or "reopened" in summary:
                reopened_tickets += 1
            if issue_type == "bug":
                bug_count += 1
            if "scope" in summary or "change request" in summary:
                scope_changes += 1

        sprint_velocity = story_points / max(lookback_days / 7, 1)
        return {
            "source": "jira",
            "project_key": self.project_key,
            "story_points_completed": story_points,
            "sprint_velocity": round(sprint_velocity, 2),
            "blocked_issues": blocked_issues,
            "reopened_tickets": reopened_tickets,
            "bug_count": bug_count,
            "scope_changes": scope_changes,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
