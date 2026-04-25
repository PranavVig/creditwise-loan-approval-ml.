import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_COLS,
    ID_COL,
    NUMERICAL_COLS,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_COL,
    ensure_dirs,
)


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    source_path = path if path else RAW_DIR / "loan_applications_sample.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {source_path}. "
            "Add your file to data/raw/loan_applications_sample.csv"
        )
    return pd.read_csv(source_path)


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )


def split_dataset(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def save_split_data(X_train, X_test, y_train, y_test) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_test.csv", index=False)


def save_feature_schema() -> None:
    schema = {
        "numerical_columns": NUMERICAL_COLS,
        "categorical_columns": CATEGORICAL_COLS,
        "target_column": TARGET_COL,
    }
    with open(PROCESSED_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def main() -> None:
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    save_split_data(X_train, X_test, y_train, y_test)
    save_feature_schema()
    print("Preprocessing split complete. Files saved to data/processed/")


if __name__ == "__main__":
    main()
