import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET_COL, ensure_dirs
from .evaluate import classification_metrics, fairness_by_group
from .preprocess import build_preprocessor


def load_processed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET_COL]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET_COL]
    return X_train, X_test, y_train, y_test


def tune_random_forest():
    ensure_dirs()
    X_train, X_test, y_train, y_test = load_processed()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    grid = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [8, 10, 12, None],
        "model__min_samples_split": [2, 5, 10],
    }
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_prob)
    metrics["best_params"] = search.best_params_

    fairness_gender = fairness_by_group(X_test.reset_index(drop=True), y_test, y_pred, "Gender")
    fairness_area = fairness_by_group(
        X_test.reset_index(drop=True), y_test, y_pred, "Property_Area"
    )

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    with open(REPORTS_DIR / "best_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(REPORTS_DIR / "fairness_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"by_gender": fairness_gender, "by_property_area": fairness_area},
            f,
            indent=2,
            default=str,
        )

    print("Saved tuned model to models/best_model.pkl")
    print("Saved best model metrics to reports/best_model_metrics.json")
    print("Saved fairness report to reports/fairness_report.json")


if __name__ == "__main__":
    tune_random_forest()
