"""
generate_charts.py
Generates 5 publication-quality charts saved to charts/
Run: python src/generate_charts.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'figure.dpi':         150,
})

COLORS = {'High': '#E24B4A', 'Medium': '#EF9F27', 'Low': '#3B9C6E'}


def chart_risk_distribution(df):
    counts = df['risk_level'].value_counts().reindex(['High', 'Medium', 'Low'])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS[l] for l in counts.index], width=0.5)
    ax.set_title('Project Risk Distribution', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Number of Projects')
    ax.set_xlabel('Risk Level')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('charts/01_risk_distribution.png')
    plt.close()
    print("Saved: charts/01_risk_distribution.png")


def chart_feature_importance(fi):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#E24B4A' if i < 3 else '#378ADD' for i in range(len(fi))]
    ax.barh(fi['feature'][::-1], fi['importance'][::-1], color=colors[::-1], height=0.6)
    ax.set_title('What Drives Project Delays?', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Feature Importance')
    red_patch   = mpatches.Patch(color='#E24B4A', label='Top 3 drivers')
    blue_patch  = mpatches.Patch(color='#378ADD', label='Supporting features')
    ax.legend(handles=[red_patch, blue_patch], loc='lower right')
    plt.tight_layout()
    plt.savefig('charts/02_feature_importance.png')
    plt.close()
    print("Saved: charts/02_feature_importance.png")


def chart_scatter_risk(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for level, color in COLORS.items():
        sub = df[df['risk_level'] == level]
        ax.scatter(sub['completion_gap'], sub['risk_score'],
                   c=color, alpha=0.65, s=40, label=level)
    ax.axhline(0.60, color='#E24B4A', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(df['completion_gap'].max() - 0.05, 0.62, 'Alert threshold',
            color='#E24B4A', fontsize=9, ha='right')
    ax.set_title('Completion Gap vs Risk Score', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Completion Gap (negative = behind schedule)')
    ax.set_ylabel('Risk Score')
    ax.legend(title='Risk Level')
    plt.tight_layout()
    plt.savefig('charts/03_scatter_risk.png')
    plt.close()
    print("Saved: charts/03_scatter_risk.png")


def chart_backtest(timeline):
    fig, ax = plt.subplots(figsize=(8, 4))
    weeks = timeline['week']
    scores = timeline['risk_score']

    # Color-coded area under curve
    ax.fill_between(weeks, scores, alpha=0.15, color='#E24B4A')
    ax.plot(weeks, scores, color='#E24B4A', linewidth=2.5, marker='o', markersize=5)

    # Alert threshold line
    ax.axhline(0.60, color='#E24B4A', linestyle='--', linewidth=1.2, alpha=0.7,
               label='Alert threshold (0.60)')

    # Find first alert week
    alert_week = timeline[timeline['risk_score'] >= 0.60]['week'].min()
    alert_score = timeline.loc[timeline['week'] == alert_week, 'risk_score'].values[0]
    ax.annotate(f'⚠ Model alert\nWeek {alert_week}',
                xy=(alert_week, alert_score),
                xytext=(alert_week + 0.8, alert_score + 0.08),
                fontsize=9, color='#E24B4A', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E24B4A', lw=1.5))

    # Mark the actual delay
    ax.axvline(10, color='#888', linestyle=':', linewidth=1.2, alpha=0.8)
    ax.text(10.1, 0.05, 'Actual delay\nconfirmed', fontsize=8.5, color='#555')

    # Shade the early warning zone
    ax.axvspan(alert_week, 10, alpha=0.06, color='#EF9F27', label=f'{10 - alert_week}-week warning window')

    ax.set_title('Risk Score Over Time — Project #47\n(Model caught delay early)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Project Week')
    ax.set_ylabel('Delay Risk Score')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(weeks)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('charts/04_backtest_timeline.png')
    plt.close()
    print("Saved: charts/04_backtest_timeline.png")


def chart_delay_rate_by_scope(df):
    rate = df.groupby('scope_changes')['delayed'].mean().reset_index()
    rate.columns = ['scope_changes', 'delay_rate']
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ['#3B9C6E' if r < 0.4 else '#EF9F27' if r < 0.65 else '#E24B4A'
                  for r in rate['delay_rate']]
    ax.bar(rate['scope_changes'], rate['delay_rate'] * 100, color=bar_colors, width=0.7)
    ax.set_title('Delay Rate by Number of Scope Changes', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Scope Changes')
    ax.set_ylabel('Delay Rate (%)')
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('charts/05_scope_vs_delay.png')
    plt.close()
    print("Saved: charts/05_scope_vs_delay.png")


if __name__ == '__main__':
    df       = pd.read_csv('data/projects_scored.csv')
    fi       = pd.read_csv('data/feature_importances.csv')
    timeline = pd.read_csv('data/backtest_timeline.csv')

    chart_risk_distribution(df)
    chart_feature_importance(fi)
    chart_scatter_risk(df)
    chart_backtest(timeline)
    chart_delay_rate_by_scope(df)

    print("\nAll 5 charts generated in charts/")
