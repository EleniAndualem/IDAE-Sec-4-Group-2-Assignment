"""Visualization utilities for EDA and model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.utils import FIGURES_DIR, TARGET_COLUMN, ensure_output_dirs

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


def _save_fig(name: str, subdir: str = "") -> Path:
    """Save current matplotlib figure to outputs/figures."""
    ensure_output_dirs()
    folder = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_histogram(df: pd.DataFrame, column: str, title: str, filename: str) -> Path:
    """Plot histogram for a numeric column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    return _save_fig(filename)


def plot_price_distribution(df: pd.DataFrame) -> Path:
    """Plot house value distribution histogram."""
    return plot_histogram(df, TARGET_COLUMN, "House Value Distribution", "house_value_distribution")


def plot_age_distribution(df: pd.DataFrame) -> Path:
    """Plot housing median age distribution histogram."""
    return plot_histogram(df, "housing_median_age", "Housing Age Distribution", "house_age_distribution")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Plot correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap")
    return _save_fig("correlation_heatmap")


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: str) -> Path:
    """Create scatter plot between two columns."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
    plt.title(title)
    return _save_fig(filename)


def plot_geographical_scatter(df: pd.DataFrame) -> Path:
    """Create a geographical scatter plot for housing data."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x="longitude", y="latitude", 
        hue=TARGET_COLUMN, size="population", sizes=(10, 200), alpha=0.5, palette="viridis"
    )
    plt.title("Geographical Distribution of House Values")
    return _save_fig("geographical_scatter")


def plot_pairplot(df: pd.DataFrame, columns: list[str] | None = None) -> Path:
    """Create pairplot for selected numeric columns."""
    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols and len(numeric_cols) > 5:
        key_cols = [TARGET_COLUMN, "median_income", "total_rooms", "housing_median_age"]
        numeric_cols = [c for c in key_cols if c in df.columns]
    sample = df[numeric_cols].dropna()
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)
    g = sns.pairplot(sample, diag_kind="kde")
    g.fig.suptitle("Pairplot of Key Numeric Features", y=1.02)
    ensure_output_dirs()
    path = FIGURES_DIR / "pairplot.png"
    g.savefig(path, bbox_inches="tight")
    plt.close("all")
    return path


def plot_ocean_proximity_distribution(df: pd.DataFrame) -> Path:
    """Plot ocean proximity category counts."""
    plt.figure(figsize=(10, 6))
    counts = df["ocean_proximity"].fillna("unknown").value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Ocean Proximity Distribution")
    plt.ylabel("Count")
    return _save_fig("ocean_proximity_distribution")


def plot_avg_value_by_ocean_proximity(df: pd.DataFrame) -> Path:
    """Plot average house value by ocean proximity."""
    avg = df.groupby("ocean_proximity")[TARGET_COLUMN].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg.index, y=avg.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average House Value by Ocean Proximity")
    plt.ylabel("Average House Value ($)")
    return _save_fig("avg_value_by_ocean_proximity")


# --- Model Evaluation Plots (Agnostic) ---

def plot_model_comparison(metrics_df: pd.DataFrame) -> Path:
    """Bar chart comparing model RMSE."""
    plt.figure(figsize=(12, 6))
    sorted_df = metrics_df.sort_values("rmse")
    sns.barplot(data=sorted_df, x="model", y="rmse", hue="model", legend=False, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Model Comparison — RMSE (lower is better)")
    plt.ylabel("RMSE ($)")
    return _save_fig("model_comparison_rmse", subdir="evaluation")


def plot_leaderboard(metrics_df: pd.DataFrame) -> Path:
    """Horizontal leaderboard of models by composite score."""
    df = metrics_df.copy()
    df["composite_score"] = (
        df["rmse"].rank()
        + df["mae"].rank()
        + df["r2"].rank(ascending=False)
        + df["train_time_sec"].rank()
    ) / 4
    df = df.sort_values("composite_score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, y="model", x="composite_score", hue="model", legend=False, palette="RdYlGn_r")
    plt.title("Model Leaderboard (lower composite score is better)")
    plt.xlabel("Composite Rank Score")
    return _save_fig("model_leaderboard", subdir="evaluation")


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
) -> Path | None:
    """Plot feature importance for tree-based models."""
    estimator = model.named_steps.get("regressor", model)
    if not hasattr(estimator, "feature_importances_"):
        return None

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_values, y=top_names, orient="h")
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance")
    safe_name = model_name.lower().replace(" ", "_")
    return _save_fig(f"feature_importance_{safe_name}", subdir="evaluation")


def plot_coefficients(
    model: Any,
    feature_names: list[str],
    model_name: str,
) -> Path | None:
    """Plot coefficients for linear models."""
    estimator = model.named_steps.get("regressor", model)
    if not hasattr(estimator, "coef_"):
        return None

    coefs = np.abs(estimator.coef_.ravel())
    indices = np.argsort(coefs)[::-1][:20]
    top_names = [feature_names[i] for i in indices]
    top_values = estimator.coef_.ravel()[indices]

    plt.figure(figsize=(10, 8))
    colors = ["green" if v >= 0 else "red" for v in top_values]
    plt.barh(top_names, top_values, color=colors)
    plt.gca().invert_yaxis()
    plt.title(f"Top Coefficients — {model_name}")
    plt.xlabel("Coefficient Value")
    safe_name = model_name.lower().replace(" ", "_")
    return _save_fig(f"coefficients_{safe_name}", subdir="evaluation")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Path:
    """Plot residual distribution and residuals vs predicted."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title(f"Residual Distribution — {model_name}")
    axes[0].set_xlabel("Residual ($)")

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.4, ax=axes[1])
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title(f"Residuals vs Predicted — {model_name}")
    axes[1].set_xlabel("Predicted House Value ($)")
    axes[1].set_ylabel("Residual ($)")

    ensure_output_dirs()
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / "evaluation" / f"residuals_{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Path:
    """Plot predicted vs actual values."""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("Actual House Value ($)")
    plt.ylabel("Predicted House Value ($)")
    plt.title(f"Predicted vs Actual — {model_name}")
    plt.legend()
    safe_name = model_name.lower().replace(" ", "_")
    return _save_fig(f"pred_vs_actual_{safe_name}", subdir="evaluation")


def plot_learning_curve(scores: dict[str, list[float]], model_name: str) -> Path:
    """Plot learning curve from cross-validation scores."""
    plt.figure(figsize=(10, 6))
    train_sizes = scores.get("train_sizes", list(range(1, len(scores["train_scores"]) + 1)))
    plt.plot(train_sizes, scores["train_scores"], label="Train R²", marker="o")
    plt.plot(train_sizes, scores["val_scores"], label="Validation R²", marker="o")
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.title(f"Learning Curve — {model_name}")
    plt.legend()
    safe_name = model_name.lower().replace(" ", "_")
    return _save_fig(f"learning_curve_{safe_name}", subdir="evaluation")


def generate_all_eda_figures(df: pd.DataFrame) -> list[Path]:
    """Generate and save all EDA figures."""
    paths: list[Path] = []
    
    if TARGET_COLUMN in df.columns:
        paths.append(plot_price_distribution(df))
        
    if "housing_median_age" in df.columns:
        paths.append(plot_age_distribution(df))
        
    paths.append(plot_correlation_heatmap(df))
    
    if "longitude" in df.columns and "latitude" in df.columns:
        paths.append(plot_geographical_scatter(df))
        
    if "median_income" in df.columns and TARGET_COLUMN in df.columns:
        paths.append(plot_scatter(df, "median_income", TARGET_COLUMN, "Income vs House Value", "income_vs_value"))
        
    paths.append(plot_pairplot(df))
    
    if "ocean_proximity" in df.columns:
        paths.append(plot_ocean_proximity_distribution(df))
        paths.append(plot_avg_value_by_ocean_proximity(df))
        
    return paths