# California Housing Prices — Streamlit app

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import engineer_features
from src.utils import (
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    descriptive_statistics,
    inspect_data,
    load_data,
)

EDA_FIGURES = [
    "house_value_distribution.png",
    "house_age_distribution.png",
    "correlation_heatmap.png",
    "geographical_scatter.png",
    "income_vs_value.png",
    "pairplot.png",
    "ocean_proximity_distribution.png",
    "avg_value_by_ocean_proximity.png",
]

OCEAN_OPTIONS = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]


@st.cache_resource
def load_model_artifacts():
    # Load saved model and feature metadata at startup
    model_path = MODELS_DIR / "best_model.pkl"
    info_path = MODELS_DIR / "feature_info.json"

    if not model_path.exists():
        st.error("Model not found. Run: `python -m src.evaluate`")
        st.stop()

    model = joblib.load(model_path)
    feature_info = json.loads(info_path.read_text(encoding="utf-8"))
    return model, feature_info


@st.cache_data
def load_dataset():
    return load_data()


@st.cache_data
def load_metrics():
    path = METRICS_DIR / "comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def page_home():
    st.title("California Housing Price Prediction")
    st.markdown(
        """
        This app predicts **median house values** in California using machine learning.

        **Workflow:** data understanding → cleaning → feature engineering → model training → evaluation → prediction
        """
    )

    df = load_dataset()
    metrics = load_metrics()
    _, feature_info = load_model_artifacts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Features", len(df.columns))
    col3.metric("Best Model", feature_info.get("model_name", "N/A"))

    if not metrics.empty:
        best_name = feature_info.get("model_name")
        best_row = metrics.loc[metrics["model"] == best_name].iloc[0]
        col4.metric("Best R²", f"{best_row['r2']:.3f}")
        st.markdown("### Best Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"${best_row['rmse']:,.0f}")
        m2.metric("MAE", f"${best_row['mae']:,.0f}")
        m3.metric("Train Time", f"{best_row['train_time_sec']:.2f}s")


def page_dataset():
    st.title("Dataset Overview")
    df = load_dataset()
    info = inspect_data(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{info['shape'][0]:,}")
    c2.metric("Columns", info["shape"][1])
    c3.metric("Duplicates", info["duplicate_rows"])

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Types & Missing Values")
    overview = pd.DataFrame({
        "column": info["columns"],
        "dtype": [info["dtypes"][c] for c in info["columns"]],
        "missing": [info["missing_values"][c] for c in info["columns"]],
    })
    st.dataframe(overview, use_container_width=True)

    st.subheader("Descriptive Statistics")
    stats = descriptive_statistics(df)
    st.dataframe(pd.DataFrame(stats["numeric"]).T, use_container_width=True)


def page_eda():
    st.title("Exploratory Data Analysis")
    st.markdown("Saved figures from the EDA pipeline.")

    for filename in EDA_FIGURES:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), caption=filename.replace("_", " ").replace(".png", "").title())
        else:
            st.warning(f"Missing figure: {filename}")


def page_model_comparison():
    st.title("Model Comparison")
    metrics = load_metrics()
    if metrics.empty:
        st.warning("No metrics found. Run: `python -m src.evaluate`")
        return

    _, feature_info = load_model_artifacts()
    best_name = feature_info.get("model_name")

    st.subheader("Metrics Table")
    display_df = metrics.copy()
    display_df["rmse"] = display_df["rmse"].map(lambda x: f"${x:,.0f}")
    display_df["mae"] = display_df["mae"].map(lambda x: f"${x:,.0f}")
    display_df["r2"] = display_df["r2"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Charts")
    col1, col2 = st.columns(2)
    rmse_chart = FIGURES_DIR / "evaluation" / "model_comparison_rmse.png"
    leaderboard = FIGURES_DIR / "evaluation" / "model_leaderboard.png"
    if rmse_chart.exists():
        col1.image(str(rmse_chart), caption="RMSE Comparison")
    if leaderboard.exists():
        col2.image(str(leaderboard), caption="Model Leaderboard")

    st.success(f"Selected best model: **{best_name}**")


def page_predict():
    st.title("Predict House Price")
    model, feature_info = load_model_artifacts()
    feature_names = feature_info["features"]

    st.markdown("Enter property details to get a price prediction.")

    col1, col2 = st.columns(2)
    with col1:
        longitude = st.number_input("Longitude", value=-118.0, format="%.4f")
        latitude = st.number_input("Latitude", value=34.0, format="%.4f")
        housing_median_age = st.number_input("Housing Median Age", value=28.0, min_value=1.0)
        total_rooms = st.number_input("Total Rooms", value=2000.0, min_value=1.0)
        total_bedrooms = st.number_input("Total Bedrooms", value=400.0, min_value=1.0)
    with col2:
        population = st.number_input("Population", value=1200.0, min_value=1.0)
        households = st.number_input("Households", value=400.0, min_value=1.0)
        median_income = st.number_input("Median Income (tens of thousands)", value=3.5, min_value=0.1)
        ocean_proximity = st.selectbox("Ocean Proximity", OCEAN_OPTIONS)

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }])

        featured = engineer_features(input_df)
        X = featured[feature_names]
        prediction = model.predict(X)[0]

        st.success(f"Predicted Median House Value: **${prediction:,.0f}**")
        with st.expander("Engineered features used"):
            st.dataframe(featured[feature_names], use_container_width=True)


def page_about():
    st.title("About")
    st.markdown(
        """
        ### IDAE Group 2 — California Housing Prices

        **Goal:** Predict median house values using regression models.

        **Dataset:** California Housing Prices (`data/California Housing Prices.csv`)

        **Pipeline modules:**
        - `src/utils.py` — load & explore data
        - `src/preprocessing.py` — clean data
        - `src/feature_engineering.py` — `houseAgeLabel`, `valuePerRoom`
        - `src/train.py` — train 7 regressors
        - `src/evaluate.py` — metrics & model selection
        - `src/visualization.py` — EDA & evaluation plots

        **Models compared:** Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR

        **Run the app:**
        ```bash
        streamlit run app/app.py
        ```
        """
    )


def main():
    st.set_page_config(
        page_title="California Housing Prices",
        page_icon="🏠",
        layout="wide",
    )

    pages = {
        "Home": page_home,
        "Dataset Overview": page_dataset,
        "EDA": page_eda,
        "Model Comparison": page_model_comparison,
        "Predict Price": page_predict,
        "About": page_about,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()


if __name__ == "__main__":
    main()
