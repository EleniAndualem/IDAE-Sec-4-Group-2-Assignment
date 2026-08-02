"""Model evaluation metrics and comparison utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve
import joblib
import json

from src.utils import (
    METRICS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
    save_markdown,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float = 0.0,
    predict_time: float = 0.0,
) -> dict[str, float]:
    """
    Compute MAE, MSE, RMSE, and R² for regression predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predictions.
    train_time : float
        Training duration in seconds.
    predict_time : float
        Prediction duration in seconds.

    Returns
    -------
    dict
        Metric name to value mapping.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
        "train_time_sec": float(train_time),
        "predict_time_sec": float(predict_time),
    }


def evaluate_model(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> tuple[dict[str, float], Any, np.ndarray]:
    """
    Train a pipeline and evaluate on the test set.

    Returns metrics dict, fitted pipeline, and test predictions.
    """
    start_train = time.perf_counter()
    fitted = pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_pred = fitted.predict(X_test)
    predict_time = time.perf_counter() - start_pred

    metrics = compute_regression_metrics(y_test.values, y_pred, train_time, predict_time)
    metrics["model"] = model_name
    return metrics, fitted, y_pred


def run_cross_validation(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "r2",
) -> dict[str, Any]:
    """
    Run k-fold cross-validation and return scores.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Model pipeline to evaluate.
    X, y : features and target
    n_splits : int
        Number of CV folds.
    scoring : str
        Sklearn scoring metric name.

    Returns
    -------
    dict
        Mean/std scores and fold-wise results.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "scoring": scoring,
        "n_splits": n_splits,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "fold_scores": scores.tolist(),
    }


def compute_learning_curve_data(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute learning curve train/validation R² scores."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    # pyrefly: ignore [bad-unpacking]
    train_sizes_abs, train_scores, val_scores, *_ = learning_curve(
        pipeline,
        X,
        y,
        train_sizes=train_sizes,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        return_times=False,
    )
    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores": train_scores.mean(axis=1).tolist(),
        "val_scores": val_scores.mean(axis=1).tolist(),
    }


def select_best_model(metrics_df: pd.DataFrame) -> str:
    """
    Select best model using composite ranking.

    Prefers lowest RMSE and MAE while penalizing slow training time.
    Does not rely on R² alone.
    """
    df = metrics_df.copy()
    df["rmse_rank"] = df["rmse"].rank()
    df["mae_rank"] = df["mae"].rank()
    df["r2_rank"] = df["r2"].rank(ascending=False)
    df["time_rank"] = df["train_time_sec"].rank()
    df["composite"] = (
        0.40 * df["rmse_rank"]
        + 0.35 * df["mae_rank"]
        + 0.15 * df["r2_rank"]
        + 0.10 * df["time_rank"]
    )
    best_row = df.loc[df["composite"].idxmin()]
    return str(best_row["model"])


def save_comparison_results(metrics_df: pd.DataFrame) -> Path:
    """Save model comparison CSV to outputs/metrics/comparison.csv."""
    ensure_output_dirs()
    path = METRICS_DIR / "comparison.csv"
    metrics_df.to_csv(path, index=False)
    return path


def generate_comparison_report(metrics_df: pd.DataFrame, best_model: str) -> str:
    """Generate markdown comparison summary."""
    ensure_output_dirs()
    sorted_df = metrics_df.sort_values("rmse")

    lines = [
        "# Model Comparison Report",
        "",
        f"**Best Model:** {best_model}",
        "",
        "## Selection Criteria",
        "",
        "The best model is selected using a composite score that weights:",
        "- RMSE (40%)",
        "- MAE (35%)",
        "- R² (15%)",
        "- Training time (10%)",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | RMSE | MAE | R² | Train Time (s) | Predict Time (s) |",
        "|------|-------|------|-----|----|----------------|------------------|",
    ]

    for rank, (_, row) in enumerate(sorted_df.iterrows(), start=1):
        marker = " ⭐" if row["model"] == best_model else ""
        lines.append(
            f"| {rank} | {row['model']}{marker} | "
            f"${row['rmse']:,.0f} | ${row['mae']:,.0f} | "
            f"{row['r2']:.4f} | {row['train_time_sec']:.3f} | "
            f"{row['predict_time_sec']:.4f} |"
        )

    best_row = metrics_df.loc[metrics_df["model"] == best_model].iloc[0]
    lines.extend([
        "",
        "## Summary",
        "",
        f"The **{best_model}** model achieved the best balance of accuracy "
        f"(RMSE=${best_row['rmse']:,.0f}, MAE=${best_row['mae']:,.0f}, R²={best_row['r2']:.4f}) "
        f"and training efficiency among all evaluated algorithms.",
        "",
    ])

    content = "\n".join(lines)
    save_markdown(content, REPORTS_DIR / "model_comparison.md")
    return content


def save_best_model(
    results_dict: dict[str, Any],
    best_model_name: str,
    feature_names: list[str],
) -> Path:
    """Save the best model and metadata to models/."""
    ensure_output_dirs()
    # The pipeline is stored under the "pipeline" key in the results dict
    model = results_dict[best_model_name]["pipeline"]
    model_path = MODELS_DIR / "best_model.pkl"

    metadata = {
        "model_name": best_model_name,
        "target_column": TARGET_COLUMN,
        "features": feature_names,
    }

    # Save as models/best_model.pkl with Joblib
    joblib.dump(model, model_path)
    
    # Save feature metadata to models/feature_info.json (as per project guide)
    feature_info_path = MODELS_DIR / "feature_info.json"
    with open(feature_info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Best model ({best_model_name}) saved to {model_path}")
    return model_path


def save_cross_validation_results(cv_results: dict[str, dict[str, Any]]) -> Path:
    # Save 5-fold CV scores to JSON
    ensure_output_dirs()
    path = METRICS_DIR / "cross_validation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=4)
    return path


def run_all_cross_validation(
    results: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> dict[str, dict[str, Any]]:
    # Run 5-fold CV for every trained model
    cv_results: dict[str, dict[str, Any]] = {}
    for name, info in results.items():
        print(f"  Cross-validating {name}...")
        cv_results[name] = run_cross_validation(
            info["pipeline"], X_train, y_train, n_splits=n_splits
        )
    return cv_results


def compute_all_learning_curves(
    results: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, dict[str, list[float]]]:
    # Learning curve data for each model
    curve_data: dict[str, dict[str, list[float]]] = {}
    for name, info in results.items():
        print(f"  Learning curve for {name}...")
        curve_data[name] = compute_learning_curve_data(info["pipeline"], X_train, y_train)
    return curve_data


if __name__ == "__main__":
    from src.train import run_training_pipeline
    from src.visualization import (
        generate_bonus_plots,
        generate_feature_importance_plots,
        plot_leaderboard,
        plot_model_comparison,
    )

    # 1. Train models and get data
    print("Running training pipeline to get models for evaluation...")
    results, X_train, X_test, y_train, y_test = run_training_pipeline()

    # 2. Evaluate each model
    print("\nEvaluating models on test set...")
    metrics_list = []
    for name, info in results.items():
        print(f"  Evaluating {name}...")
        pipeline = info["pipeline"]
        train_time = info["train_time_sec"]

        start_pred = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        predict_time = time.perf_counter() - start_pred

        metrics = compute_regression_metrics(y_test.values, y_pred, train_time, predict_time)
        metrics["model"] = name
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # 3. Save comparison results (Step 8)
    csv_path = save_comparison_results(metrics_df)
    print(f"\nSaved metrics to {csv_path}")

    # 4. Compare and select best model (Step 9)
    best_model = select_best_model(metrics_df)
    print(f"Selected Best Model: {best_model}")

    # 4b. Save the best model (Step 10)
    save_best_model(results, best_model, X_train.columns.tolist())

    # 5. Generate report and visualizations (Step 9)
    generate_comparison_report(metrics_df, best_model)
    print("Generated model comparison markdown report.")

    plot_model_comparison(metrics_df)
    plot_leaderboard(metrics_df)
    print("Generated model comparison visualizations.")

    # 6. Feature importance / coefficients (Step 11)
    print("\nGenerating feature importance and coefficient plots...")
    importance_paths = generate_feature_importance_plots(results, X_train)
    print(f"Saved {len(importance_paths)} importance/coefficient plots.")

    # 7. Bonus analysis (Step 12)
    print("\nRunning 5-fold cross-validation...")
    cv_results = run_all_cross_validation(results, X_train, y_train)
    cv_path = save_cross_validation_results(cv_results)
    print(f"Saved cross-validation results to {cv_path}")

    print("\nComputing learning curves...")
    learning_curve_data = compute_all_learning_curves(results, X_train, y_train)

    print("\nGenerating residual, prediction, and learning curve plots...")
    bonus_paths = generate_bonus_plots(
        results, X_train, X_test, y_test, learning_curve_data
    )
    print(f"Saved {len(bonus_paths)} bonus analysis plots.")