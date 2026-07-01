"""
github_connector.py
GitHub REST API connector for engineering activity metrics.
Credentials:
  GITHUB_TOKEN optional but recommended for higher rate limits.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests


class GitHubConnector:
    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url.rstrip("/")

    @property
    def headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, params: Optional[Dict[str, object]] = None) -> object:
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self.headers,
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def fetch_repo_metrics(self, owner: str, repo: str, lookback_days: int = 7) -> Dict[str, object]:
        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        commits = self._get(f"/repos/{owner}/{repo}/commits", {"since": since, "per_page": 100})
        open_prs = self._get(f"/repos/{owner}/{repo}/pulls", {"state": "open", "per_page": 100})
        closed_prs = self._get(f"/repos/{owner}/{repo}/pulls", {"state": "closed", "per_page": 100})
        issues = self._get(
            f"/repos/{owner}/{repo}/issues",
            {"state": "all", "since": since, "per_page": 100},
        )

        merged_prs = [pr for pr in closed_prs if pr.get("merged_at")]
        contributors = {
            commit.get("author", {}).get("login")
            for commit in commits
            if commit.get("author", {}).get("login")
        }

        issue_items = [issue for issue in issues if "pull_request" not in issue]
        closed_issues = [issue for issue in issue_items if issue.get("closed_at")]

        return {
            "source": "github",
            "repository": f"{owner}/{repo}",
            "commits_per_week": len(commits),
            "pull_requests_opened": self._count_recent_prs(open_prs + closed_prs, since),
            "open_pull_requests": len(open_prs),
            "pull_requests_merged": len(merged_prs),
            "average_merge_time_hours": self._average_merge_time_hours(merged_prs),
            "number_of_contributors": len(contributors),
            "issue_creation_rate": len(issue_items),
            "issue_closure_rate": len(closed_issues),
            "code_churn": self._estimate_code_churn(commits),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _count_recent_prs(prs: List[Dict[str, object]], since: str) -> int:
        since_dt = datetime.fromisoformat(since)
        count = 0
        for pr in prs:
            created_at = pr.get("created_at")
            if created_at and datetime.fromisoformat(created_at.replace("Z", "+00:00")) >= since_dt:
                count += 1
        return count

    @staticmethod
    def _average_merge_time_hours(merged_prs: List[Dict[str, object]]) -> float:
        durations = []
        for pr in merged_prs:
            created = pr.get("created_at")
            merged = pr.get("merged_at")
            if created and merged:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                merged_dt = datetime.fromisoformat(merged.replace("Z", "+00:00"))
                durations.append((merged_dt - created_dt).total_seconds() / 3600)
        return round(sum(durations) / len(durations), 2) if durations else 0.0

    @staticmethod
    def _estimate_code_churn(commits: List[Dict[str, object]]) -> int:
        # Commit-list responses do not include per-file stats. This proxy is stable for dashboards;
        # a production pipeline can enrich each commit via /commits/{sha}.
        return len(commits) * 25


