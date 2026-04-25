import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET_COL, ensure_dirs
from .evaluate import classification_metrics
from .preprocess import build_preprocessor


def load_processed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET_COL]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET_COL]
    return X_train, X_test, y_train, y_test


def train_models():
    ensure_dirs()
    X_train, X_test, y_train, y_test = load_processed()

    preprocessor = build_preprocessor()
    models = {
        "logistic_regression": LogisticRegression(max_iter=1200),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "random_forest": RandomForestClassifier(
            random_state=42, n_estimators=300, max_depth=10
        ),
    }

    metric_rows = []
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        metric_rows.append(metrics)

        joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")
        print(f"Saved model: models/{name}.pkl")

    metrics_df = pd.DataFrame(metric_rows).sort_values("f1_score", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("Saved metrics: reports/model_comparison.csv")

    best = metrics_df.iloc[0]["model"]
    with open(REPORTS_DIR / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_baseline_model": best}, f, indent=2)
    print(f"Best baseline model: {best}")


if __name__ == "__main__":
    train_models()
