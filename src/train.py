# Train all regression models on California housing data

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from feature_engineering import engineer_features
from preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    create_clean_dataframe,
)
from utils import load_data

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model name → (regressor instance, whether to scale numeric features)
MODEL_CONFIGS: dict[str, tuple[object, bool]] = {
    "Linear Regression": (LinearRegression(), True),
    "Ridge Regression": (Ridge(alpha=1.0, random_state=RANDOM_STATE), True),
    "Lasso Regression": (Lasso(alpha=0.1, random_state=RANDOM_STATE), True),
    "Decision Tree": (DecisionTreeRegressor(random_state=RANDOM_STATE), False),
    "Random Forest": (
        RandomForestRegressor(
            n_estimators=30,
            max_depth=20,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=RANDOM_STATE),
        False,
    ),
    "SVR": (SVR(kernel="rbf"), True),
}


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Split columns into numeric and categorical, excluding target
    numeric_features = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != TARGET_COLUMN
    ]
    categorical_features = [
        col for col in df.select_dtypes(include=["object", "string", "category"]).columns
    ]
    return numeric_features, categorical_features


def build_model_pipeline(
    regressor: object,
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> Pipeline:
    # Combine preprocessor and regressor into a single pipeline
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Load, clean, engineer features, and split 80/20
    raw_df = load_data()
    clean_df = create_clean_dataframe(raw_df)
    featured_df = engineer_features(clean_df)

    X = featured_df.drop(columns=[TARGET_COLUMN])
    y = featured_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, dict]:
    # Train all 7 models, return fitted pipelines and training times
    numeric_features, categorical_features = get_feature_columns(X_train)
    results: dict[str, dict] = {}

    for name, (regressor, scale) in MODEL_CONFIGS.items():
        print(f"Training {name}...")
        pipeline = build_model_pipeline(regressor, numeric_features, categorical_features, scale)

        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        results[name] = {
            "pipeline": pipeline,
            "train_time_sec": round(train_time, 4),
        }
        print(f"  Done in {train_time:.4f}s")

    return results


def run_training_pipeline() -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Full pipeline: prepare data → train all models → return everything
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows\n")

    results = train_all_models(X_train, y_train)

    print(f"\nAll {len(results)} models trained successfully.")
    return results, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    results, X_train, X_test, y_train, y_test = run_training_pipeline()

    print("\n--- Training Summary ---")
    for name, info in results.items():
        print(f"  {name}: {info['train_time_sec']}s")
