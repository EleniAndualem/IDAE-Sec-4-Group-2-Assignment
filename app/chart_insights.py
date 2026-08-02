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
