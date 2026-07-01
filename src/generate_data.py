"""
generate_data.py
Simulates 300 realistic IT project records with 8 health signals.
Run this first: python src/generate_data.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 300

df = pd.DataFrame({
    'project_id':             range(1, N + 1),
    'team_size':              np.random.randint(3, 20, N),
    'planned_duration_days':  np.random.randint(30, 180, N),
    'sprint_velocity':        np.random.uniform(0.5, 1.5, N),   # 1.0 = on track
    'tasks_completed_pct':    np.random.uniform(20, 95, N),
    'scope_changes':          np.random.randint(0, 8, N),
    'bugs_open':              np.random.randint(0, 50, N),
    'team_availability_pct':  np.random.uniform(60, 100, N),
    'days_elapsed':           np.random.randint(10, 120, N),
    'stakeholder_changes':    np.random.randint(0, 5, N),
})

# Realistic target: delayed = 1 when multiple risk signals fire
df['delayed'] = (
    (df['sprint_velocity'] < 0.75) |
    (df['scope_changes'] > 4) |
    (df['bugs_open'] > 30) |
    (df['team_availability_pct'] < 72) |
    (df['stakeholder_changes'] > 3)
).astype(int)

# Add some noise so it's not perfectly separable
noise_idx = np.random.choice(N, size=20, replace=False)
df.loc[noise_idx, 'delayed'] = 1 - df.loc[noise_idx, 'delayed']

df.to_csv('data/projects.csv', index=False)

delayed_count = df['delayed'].sum()
print(f"Generated {N} projects")
print(f"  Delayed:    {delayed_count} ({delayed_count/N*100:.1f}%)")
print(f"  On time:    {N - delayed_count} ({(N - delayed_count)/N*100:.1f}%)")
print(f"  Saved to:   data/projects.csv")
print(f"\nFirst 5 rows:\n{df.head()}")
