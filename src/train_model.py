"""
train_model.py
Trains a Random Forest classifier, evaluates it, and exports:
  - models/delay_model.pkl
  - data/projects_scored.csv
  - data/feature_importances.csv
Run: python src/train_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.feature_engineering import engineer_features, FEATURE_COLS


def train():
    # â”€â”€ 1. Load & engineer features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = pd.read_csv('data/projects.csv')
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df['delayed']

    # â”€â”€ 2. Train / test split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # â”€â”€ 3. Train model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)

    # â”€â”€ 4. Evaluate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    preds      = model.predict(X_test)
    probs      = model.predict_proba(X_test)[:, 1]
    roc_auc    = roc_auc_score(y_test, probs)
    cv_scores  = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"\nROC-AUC (test):         {roc_auc:.3f}")
    print(f"ROC-AUC (5-fold CV):    {cv_scores.mean():.3f} Â± {cv_scores.std():.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, preds)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")

    # â”€â”€ 5. Save model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with open('models/delay_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved -> models/delay_model.pkl")

    # â”€â”€ 6. Score all projects and export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df['risk_score'] = model.predict_proba(X)[:, 1]
    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[0, 0.35, 0.65, 1.0],
        labels=['Low', 'Medium', 'High'],
    )
    df.to_csv('data/projects_scored.csv', index=False)
    df.to_json('data/projects_scored.json', orient='records', indent=2)
    print(f"Scored projects saved -> data/projects_scored.csv & .json")
    print(f"\nRisk distribution:\n{df['risk_level'].value_counts()}")

    # â”€â”€ 7. Export feature importances â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fi = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    fi.to_csv('data/feature_importances.csv', index=False)
    fi.to_json('data/feature_importances.json', orient='records', indent=2)
    print(f"Feature importances saved -> data/feature_importances.csv & .json")

    # Generate intelligence artifacts used by the upgraded dashboard panels.
    try:
        from src.explainability import generate_explanations
        from src.recommendation_engine import generate_recommendations

        generate_explanations()
        generate_recommendations()
    except Exception as exc:
        print(f"Warning: intelligence artifact generation skipped: {exc}")

    return model, df, fi


if __name__ == '__main__':
    train()

