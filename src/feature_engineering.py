"""
feature_engineering.py
Creates derived features that are more predictive than raw columns.
Import this module in other scripts: from src.feature_engineering import engineer_features
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a project DataFrame."""
    df = df.copy()

    # Velocity trend: how fast tasks are being completed per day
    df['velocity_trend'] = df['tasks_completed_pct'] / df['days_elapsed'].replace(0, 1)

    # Scope risk: multiplies scope changes by open bugs — compound risk
    df['scope_risk'] = df['scope_changes'] * df['bugs_open']

    # Team pressure: tasks done relative to available team capacity
    df['team_pressure'] = df['tasks_completed_pct'] / df['team_availability_pct'].replace(0, 1)

    # Completion gap: are we ahead or behind schedule?
    # Negative = falling behind (% work done < % time used)
    df['completion_gap'] = (
        (df['tasks_completed_pct'] / 100)
        - (df['days_elapsed'] / df['planned_duration_days'].replace(0, 1))
    )

    # Bug density: open bugs relative to team size
    df['bug_density'] = df['bugs_open'] / df['team_size'].replace(0, 1)

    return df


FEATURE_COLS = [
    'team_size',
    'sprint_velocity',
    'tasks_completed_pct',
    'scope_changes',
    'bugs_open',
    'team_availability_pct',
    'velocity_trend',
    'scope_risk',
    'completion_gap',
    'bug_density',
]

if __name__ == '__main__':
    df = pd.read_csv('data/projects.csv')
    df = engineer_features(df)
    print("Features added successfully.")
    print(df[FEATURE_COLS].describe().round(3))
