# Chart titles and interpretations for the web app

FIGURE_INSIGHTS: dict[str, dict[str, str]] = {
    "house_value_distribution.png": {
        "title": "House Value Distribution",
        "interpretation": (
            "Most districts have median house values between $100k–$300k, with a long right tail "
            "toward expensive coastal areas. The distribution is right-skewed, so a few high-value "
            "blocks pull the mean above the median."
        ),
    },
    "house_age_distribution.png": {
        "title": "Housing Age Distribution",
        "interpretation": (
            "Housing age peaks around 20–35 years, showing many mid-age buildings. Very new and very "
            "old homes are less common, which matters because age often relates to price and maintenance."
        ),
    },
    "correlation_heatmap.png": {
        "title": "Correlation Heatmap",
        "interpretation": (
            "Median income shows a strong positive correlation with house value. Room and population "
            "features are also related to each other. Location variables help capture regional price differences."
        ),
    },
    "geographical_scatter.png": {
        "title": "Geographical Price Map",
        "interpretation": (
            "Higher prices cluster near the coast and around the Bay Area / Southern California metros. "
            "Inland districts generally show lower median values, confirming location as a key driver."
        ),
    },
    "income_vs_value.png": {
        "title": "Income vs House Value",
        "interpretation": (
            "As median income rises, house values tend to rise as well. The relationship is positive "
            "but not perfectly linear because location and housing stock also influence price."
        ),
    },
    "pairplot.png": {
        "title": "Pairplot of Key Features",
        "interpretation": (
            "This view shows how the main numeric features move together. It helps spot linear trends, "
            "clusters, and outliers before modeling."
        ),
    },
    "ocean_proximity_distribution.png": {
        "title": "Ocean Proximity Distribution",
        "interpretation": (
            "Most blocks are either within 1 hour of the ocean or inland. Fewer districts are directly "
            "on the bay or ocean, so those categories can be more influential despite lower frequency."
        ),
    },
    "avg_value_by_ocean_proximity.png": {
        "title": "Average Value by Ocean Proximity",
        "interpretation": (
            "Coastal categories tend to have higher average prices than inland blocks. Ocean proximity "
            "is an important categorical signal for the regression models."
        ),
    },
    "model_comparison_rmse.png": {
        "title": "RMSE Comparison",
        "interpretation": (
            "Lower RMSE means better overall prediction accuracy in dollars. Tree ensembles perform best, "
            "while SVR struggles on this dataset with default settings."
        ),
    },
    "model_leaderboard.png": {
        "title": "Model Leaderboard",
        "interpretation": (
            "Models are ranked using a composite score: RMSE (40%), MAE (35%), R² (15%), and training time (10%). "
            "Each metric is converted to a rank (1 = best), then combined with these weights. "
            "Lower composite score wins — Random Forest ranks best overall."
        ),
    },
    # --- Model Analysis: Feature Importance (tree models) ---
    "feature_importance_random_forest.png": {
        "title": "Feature Importance — Random Forest",
        "interpretation": (
            "median_income and location (latitude, longitude) drive most splits — coastal and Bay Area "
            "blocks cost more. valuePerRoom and ocean_proximity add signal beyond raw counts, matching "
            "our best test performance (R² 0.806, RMSE $50,476)."
        ),
    },
    "feature_importance_decision_tree.png": {
        "title": "Feature Importance — Decision Tree",
        "interpretation": (
            "Same top drivers as Random Forest (income, location), but a single tree over-relies on "
            "a few splits. Test R² 0.653 — useful features, weaker generalization than the ensemble."
        ),
    },
    "feature_importance_gradient_boosting.png": {
        "title": "Feature Importance — Gradient Boosting",
        "interpretation": (
            "Income and location lead; later trees refine mid-range price errors. Strong second place "
            "(R² 0.774, RMSE $54,380) but slightly less stable than Random Forest across CV folds."
        ),
    },
    # --- Model Analysis: Linear Coefficients ---
    "coefficients_linear_regression.png": {
        "title": "Linear Coefficients — Linear Regression",
        "interpretation": (
            "median_income has the largest positive effect on price; coastal ocean_proximity categories "
            "are positive, inland lower. RMSE ~$69,944 — linear weights miss non-linear location effects."
        ),
    },
    "coefficients_ridge_regression.png": {
        "title": "Linear Coefficients — Ridge Regression",
        "interpretation": (
            "Nearly identical to Linear Regression (RMSE $69,951). L2 regularization slightly shrinks "
            "large coefficients but does not fix the model's limited flexibility on housing data."
        ),
    },
    "coefficients_lasso_regression.png": {
        "title": "Linear Coefficients — Lasso Regression",
        "interpretation": (
            "Same pattern as Linear/Ridge (RMSE $69,944). L1 penalty barely zeros features — most "
            "inputs remain useful, so Lasso offers little advantage here."
        ),
    },
    # --- Model Analysis: Learning Curves ---
    "learning_curve_random_forest.png": {
        "title": "Learning Curve — Random Forest",
        "interpretation": (
            "Validation R² climbs to ~0.80 with a small train–val gap. The model generalizes well; "
            "the curve flattens near 100% data, so more rows would help only marginally."
        ),
    },
    "learning_curve_gradient_boosting.png": {
        "title": "Learning Curve — Gradient Boosting",
        "interpretation": (
            "Validation R² reaches ~0.77–0.78. Train score stays above validation — mild overfitting "
            "at larger sample sizes, but still a strong performer."
        ),
    },
    "learning_curve_decision_tree.png": {
        "title": "Learning Curve — Decision Tree",
        "interpretation": (
            "Train R² stays high while validation lags — classic overfitting. A single tree memorizes "
            "training noise; ensemble methods (RF, GB) address this."
        ),
    },
    "learning_curve_linear_regression.png": {
        "title": "Learning Curve — Linear Regression",
        "interpretation": (
            "Train and validation both plateau near R² ~0.63. More data does not help — the model is "
            "too simple for housing's non-linear price patterns."
        ),
    },
    "learning_curve_ridge_regression.png": {
        "title": "Learning Curve — Ridge Regression",
        "interpretation": (
            "Same flat ~0.63 curves as Linear Regression. Regularization does not unlock new capacity "
            "on this dataset."
        ),
    },
    "learning_curve_lasso_regression.png": {
        "title": "Learning Curve — Lasso Regression",
        "interpretation": (
            "Same ~0.63 plateau as other linear models. Feature count was not the bottleneck — "
            "model form was."
        ),
    },
    "learning_curve_svr.png": {
        "title": "Learning Curve — SVR",
        "interpretation": (
            "Validation R² stays near or below zero at all training sizes. Default RBF SVR fails on "
            "this feature space — needs tuning or a different algorithm."
        ),
    },
    # --- Model Analysis: Residuals ---
    "residuals_random_forest.png": {
        "title": "Residual Analysis — Random Forest",
        "interpretation": (
            "Errors center near $0 (low bias). Slight funnel at high prices — expensive block groups "
            "are harder to predict. Typical error ~$33k (MAE)."
        ),
    },
    "residuals_gradient_boosting.png": {
        "title": "Residual Analysis — Gradient Boosting",
        "interpretation": (
            "Similar shape to Random Forest but slightly wider spread. Some under-prediction at the "
            "top of the price range ($400k+)."
        ),
    },
    "residuals_decision_tree.png": {
        "title": "Residual Analysis — Decision Tree",
        "interpretation": (
            "Wider error distribution than ensembles. More extreme residuals on outlier districts — "
            "single-tree predictions are less stable."
        ),
    },
    "residuals_linear_regression.png": {
        "title": "Residual Analysis — Linear Regression",
        "interpretation": (
            "Broader cloud than tree models; systematic error in high-price coastal areas. Linear "
            "weights cannot capture location × income interactions."
        ),
    },
    "residuals_ridge_regression.png": {
        "title": "Residual Analysis — Ridge Regression",
        "interpretation": (
            "Nearly identical spread to Linear Regression. Regularization did not reduce systematic "
            "bias in expensive districts."
        ),
    },
    "residuals_lasso_regression.png": {
        "title": "Residual Analysis — Lasso Regression",
        "interpretation": (
            "Same broad residual pattern as Linear/Ridge. Under-predicts expensive coastal blocks, "
            "over-predicts some cheaper inland areas."
        ),
    },
    "residuals_svr.png": {
        "title": "Residual Analysis — SVR",
        "interpretation": (
            "Large, skewed errors with no stable center at zero. Worse than predicting the average "
            "price — confirms SVR should not be deployed."
        ),
    },
    # --- Model Analysis: Predicted vs Actual ---
    "pred_vs_actual_random_forest.png": {
        "title": "Predicted vs Actual — Random Forest",
        "interpretation": (
            "Tightest cluster around the diagonal (R² 0.806). Best overall fit; most points within "
            "the ~$50k RMSE band."
        ),
    },
    "pred_vs_actual_gradient_boosting.png": {
        "title": "Predicted vs Actual — Gradient Boosting",
        "interpretation": (
            "Good alignment (R² 0.774) but more scatter than Random Forest at high actual values — "
            "expensive districts are slightly harder for this model."
        ),
    },
    "pred_vs_actual_decision_tree.png": {
        "title": "Predicted vs Actual — Decision Tree",
        "interpretation": (
            "Moderate fit (R² 0.653). Visible compression — extreme prices pulled toward the mean "
            "because one tree cannot smooth across regions as well as an ensemble."
        ),
    },
    "pred_vs_actual_linear_regression.png": {
        "title": "Predicted vs Actual — Linear Regression",
        "interpretation": (
            "Points spread far from the diagonal (R² 0.627). Under-predicts expensive districts, "
            "over-predicts some cheaper inland blocks."
        ),
    },
    "pred_vs_actual_ridge_regression.png": {
        "title": "Predicted vs Actual — Ridge Regression",
        "interpretation": (
            "Nearly identical scatter to Linear Regression (R² 0.627). Shrinking coefficients "
            "did not improve point-level accuracy."
        ),
    },
    "pred_vs_actual_lasso_regression.png": {
        "title": "Predicted vs Actual — Lasso Regression",
        "interpretation": (
            "Same wide spread as other linear models (R² 0.627). Feature selection did not close "
            "the gap to tree-based performers."
        ),
    },
    "pred_vs_actual_svr.png": {
        "title": "Predicted vs Actual — SVR",
        "interpretation": (
            "Wide scatter with poor diagonal alignment (R² −0.04). Predictions barely track actual "
            "prices — worst performer on the hold-out test set."
        ),
    },
}

CHART_PREFIX_META = {
    "feature_importance": {
        "label": "Feature Importance",
        "interpretation": (
            "Shows which inputs most influence predictions for tree-based models. Higher bars mean "
            "the feature contributes more to splitting decisions and final price estimates."
        ),
    },
    "coefficients": {
        "label": "Linear Coefficients",
        "interpretation": (
            "Shows positive and negative linear effects after preprocessing. Larger absolute values "
            "indicate stronger influence on predicted house value for linear models."
        ),
    },
    "learning_curve": {
        "label": "Learning Curve",
        "interpretation": (
            "Shows whether more training data improves validation performance. Converging train and "
            "validation curves suggest the model is learning general patterns rather than memorizing noise."
        ),
    },
    "residuals": {
        "label": "Residual Analysis",
        "interpretation": (
            "Residuals should scatter randomly around zero. Funnel shapes or curves suggest the model "
            "systematically under- or over-predicts in certain price ranges."
        ),
    },
    "pred_vs_actual": {
        "label": "Predicted vs Actual",
        "interpretation": (
            "Points close to the diagonal line indicate accurate predictions. Wider spread at high "
            "prices shows where expensive districts are harder to estimate."
        ),
    },
}


def _model_label_from_stem(stem: str, prefix: str) -> str:
    slug = stem.removeprefix(prefix).strip("_")
    return slug.replace("_", " ").title()


def get_chart_meta(filename: str) -> dict[str, str]:
    if filename in FIGURE_INSIGHTS:
        return FIGURE_INSIGHTS[filename]

    stem = filename.removesuffix(".png")
    for prefix, meta in CHART_PREFIX_META.items():
        token = f"{prefix}_"
        if stem.startswith(token):
            model = _model_label_from_stem(stem, token)
            return {
                "title": f"{meta['label']} — {model}",
                "interpretation": meta["interpretation"],
            }

    return {
        "title": stem.replace("_", " ").title(),
        "interpretation": "",
    }
