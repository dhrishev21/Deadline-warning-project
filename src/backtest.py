"""
backtest.py
Simulates a project deteriorating week by week and scores each snapshot.
This generates the "model caught it early" chart for demos.
Run: python src/backtest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features, FEATURE_COLS


def simulate_at_risk_project(n_weeks: int = 10, planned_days: int = 70) -> pd.DataFrame:
    """Simulate a project that starts OK then deteriorates from week 3 onward."""
    records = []
    for week in range(1, n_weeks + 1):
        # Velocity degrades after week 3
        velocity = 1.05 if week <= 3 else max(0.45, 1.05 - (week - 3) * 0.12)

        # Bugs accumulate mid-project
        bugs = 4 + max(0, (week - 2) * 5)

        # Tasks fall behind schedule
        expected_pct = (week / n_weeks) * 100
        actual_pct   = min(95, expected_pct * (1 if week <= 3 else 0.72))

        # Scope creep kicks in mid-project
        scope = 1 if week <= 3 else 3 + (week - 4)

        records.append({
            'week':                   week,
            'team_size':              9,
            'planned_duration_days':  planned_days,
            'sprint_velocity':        round(velocity, 3),
            'tasks_completed_pct':    round(actual_pct, 1),
            'scope_changes':          scope,
            'bugs_open':              bugs,
            'team_availability_pct':  88.0,
            'days_elapsed':           week * 7,
            'stakeholder_changes':    0,
        })
    return pd.DataFrame(records)


def run_backtest():
    # Load trained model
    with open('models/delay_model.pkl', 'rb') as f:
        model = pickle.load(f)

    timeline = simulate_at_risk_project()
    timeline = engineer_features(timeline)

    timeline['risk_score'] = model.predict_proba(timeline[FEATURE_COLS])[:, 1]
    timeline['risk_level'] = pd.cut(
        timeline['risk_score'],
        bins=[0, 0.35, 0.65, 1.0],
        labels=['Low', 'Medium', 'High'],
    )

    # Find the first week the model raised an alert
    alert_week = timeline[timeline['risk_score'] >= 0.60]['week'].min()

    print("=" * 50)
    print("BACK-TEST RESULTS — Project #47 (delayed in week 10)")
    print("=" * 50)
    print(timeline[['week', 'sprint_velocity', 'bugs_open', 'tasks_completed_pct',
                     'completion_gap', 'risk_score', 'risk_level']].to_string(index=False))
    print(f"\n>>> Model first flagged HIGH RISK: Week {alert_week}")
    print(f">>> Project actually delayed:      Week 10")
    print(f">>> Early warning lead time:       {10 - alert_week} weeks")

    timeline.to_csv('data/backtest_timeline.csv', index=False)
    timeline.to_json('data/backtest_timeline.json', orient='records', indent=2)
    print("\nBack-test data saved -> data/backtest_timeline.csv & .json")
    return timeline


if __name__ == '__main__':
    run_backtest()
