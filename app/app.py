# California Housing Prices — modern Streamlit web app

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_chart_spec = importlib.util.spec_from_file_location(
    "chart_insights",
    Path(__file__).resolve().parent / "chart_insights.py",
)
_chart_mod = importlib.util.module_from_spec(_chart_spec)
_chart_spec.loader.exec_module(_chart_mod)
get_chart_meta = _chart_mod.get_chart_meta
from src.feature_engineering import engineer_features
from src.utils import (
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    inspect_data,
    load_data,
)

OCEAN_OPTIONS = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
EVAL_DIR = FIGURES_DIR / "evaluation"
CV_PATH = METRICS_DIR / "cross_validation.json"

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }

    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(120deg, #0b2447 0%, #19376d 45%, #0f9b8e 100%);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 1.75rem;
        box-shadow: 0 18px 40px rgba(15, 36, 71, 0.25);
    }
    .hero h1 { color: #ffffff !important; margin: 0 0 0.35rem 0; font-size: 2.1rem; }
    .hero p { color: #d7f3ff !important; margin: 0; font-size: 1.05rem; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e8eef7;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(15, 36, 71, 0.06);
        color: #0f172a;
    }
    .metric-card .label { color: #64748b !important; font-size: 0.85rem; margin-bottom: 0.2rem; }
    .metric-card .value { color: #0f172a !important; font-size: 1.45rem; font-weight: 700; }

    .info-card {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
        border: 1px solid #dbeafe;
        border-left: 5px solid #0f9b8e;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        color: #1e293b;
    }
    .info-card strong { color: #0b2447 !important; }

    .interp-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        margin: 0.75rem 0 1.5rem 0;
        color: #78350f !important;
        line-height: 1.55;
    }
    .interp-box strong { color: #92400e !important; }

    .result-box {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1px solid #6ee7b7;
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        margin: 1rem 0;
        color: #065f46 !important;
    }
    .result-price { font-size: 2.4rem; font-weight: 700; color: #047857 !important; }

    .chart-wrap {
        background: #ffffff;
        border: 1px solid #e8eef7;
        border-radius: 16px;
        padding: 1rem 1rem 0.25rem 1rem;
        margin-bottom: 0.5rem;
        color: #0f172a;
    }
    .chart-wrap h4 { color: #0f172a !important; }

    .about-section {
        background: #ffffff;
        border: 1px solid #e8eef7;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        color: #1e293b;
    }
    .about-section h3 { color: #0b2447 !important; margin-top: 0; }
    .about-section ul { margin-bottom: 0; }

    /* Dark mode — explicit colors so text stays readable on cards */
    .stApp[data-theme="dark"] .metric-card {
        background: #1e293b;
        border-color: #334155;
        color: #f1f5f9;
    }
    .stApp[data-theme="dark"] .metric-card .label { color: #94a3b8 !important; }
    .stApp[data-theme="dark"] .metric-card .value { color: #f8fafc !important; }

    .stApp[data-theme="dark"] .info-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-color: #334155;
        color: #e2e8f0;
    }
    .stApp[data-theme="dark"] .info-card strong { color: #f8fafc !important; }

    .stApp[data-theme="dark"] .interp-box {
        background: #422006;
        border-color: #b45309;
        color: #fde68a !important;
    }
    .stApp[data-theme="dark"] .interp-box strong { color: #fcd34d !important; }

    .stApp[data-theme="dark"] .result-box {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border-color: #10b981;
        color: #d1fae5 !important;
    }
    .stApp[data-theme="dark"] .result-price { color: #6ee7b7 !important; }

    .stApp[data-theme="dark"] .chart-wrap {
        background: #1e293b;
        border-color: #334155;
        color: #f1f5f9;
    }
    .stApp[data-theme="dark"] .chart-wrap h4 { color: #f8fafc !important; }

    .stApp[data-theme="dark"] .about-section {
        background: #1e293b;
        border-color: #334155;
        color: #e2e8f0;
    }
    .stApp[data-theme="dark"] .about-section h3 { color: #f8fafc !important; }
</style>
"""


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="info-card"><strong>{title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )


def show_chart(path: Path, *, show_interpretation: bool = True) -> None:
    meta = get_chart_meta(path.name)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown(f"#### {meta['title']}")
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Chart not found: `{path}`")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if show_interpretation and meta.get("interpretation"):
        st.markdown(
            f'<div class="interp-box"><strong>Interpretation:</strong> {meta["interpretation"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def show_all_charts(paths: list[Path]) -> None:
    for path in paths:
        show_chart(path)


def compute_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    # Same weights as src/evaluate.py select_best_model()
    df = metrics.copy()
    df["rmse_rank"] = df["rmse"].rank()
    df["mae_rank"] = df["mae"].rank()
    df["r2_rank"] = df["r2"].rank(ascending=False)
    df["time_rank"] = df["train_time_sec"].rank()
    df["composite_score"] = (
        0.40 * df["rmse_rank"]
        + 0.35 * df["mae_rank"]
        + 0.15 * df["r2_rank"]
        + 0.10 * df["time_rank"]
    )
    df = df.sort_values("composite_score").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def about_section(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="about-section"><h3>{title}</h3>{body_html}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_artifacts():
    model_path = MODELS_DIR / "best_model.pkl"
    info_path = MODELS_DIR / "feature_info.json"
    if not model_path.exists():
        st.error("Model not found. Run: `python -m src.evaluate`")
        st.stop()
    return joblib.load(model_path), json.loads(info_path.read_text(encoding="utf-8"))


@st.cache_data
def load_dataset():
    return load_data()


@st.cache_data
def load_metrics():
    path = METRICS_DIR / "comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_cv_results():
    if not CV_PATH.exists():
        return {}
    return json.loads(CV_PATH.read_text(encoding="utf-8"))


def init_session_state():
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def page_home():
    hero("California Housing Price Predictor", "End-to-end ML pipeline for median house value estimation")
    df = load_dataset()
    metrics = load_metrics()
    _, feature_info = load_model_artifacts()
    best_name = feature_info.get("model_name", "N/A")
    best_row = metrics.loc[metrics["model"] == best_name].iloc[0] if not metrics.empty else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Districts", f"{len(df):,}")
    with c2:
        metric_card("Features", str(len(df.columns)))
    with c3:
        metric_card("Best Model", best_name)
    with c4:
        metric_card("Best R²", f"{best_row['r2']:.3f}" if best_row is not None else "—")

    st.markdown("### Project Description")
    info_card(
        "Objective",
        "Build and deploy a regression system that predicts median house values for California census "
        "block groups. The workflow covers data cleaning, feature engineering, model comparison, "
        "evaluation, and interactive prediction through this web application.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Dataset Information")
        info = inspect_data(df)
        info_card("Source", f"`{DEFAULT_DATA_PATH.name}` — 1990 California census block groups.")
        info_card("Target Variable", f"`{TARGET_COLUMN}` — median house value in USD.")
        info_card("Records", f"{info['shape'][0]:,} rows × {info['shape'][1]} columns")
        info_card("Missing Values", "None after preprocessing pipeline.")
        with st.expander("Preview dataset"):
            st.dataframe(df.head(10), use_container_width=True)

    with col_b:
        st.markdown("### Model Information")
        info_card("Selected Model", f"{best_name} — chosen by composite score (RMSE, MAE, R², speed).")
        info_card("Engineered Features", "`houseAgeLabel` (age category), `valuePerRoom` (rooms per household).")
        if best_row is not None:
            info_card(
                "Hold-out Performance",
                f"RMSE ${best_row['rmse']:,.0f} · MAE ${best_row['mae']:,.0f} · R² {best_row['r2']:.4f}",
            )
        info_card("Pipeline", "sklearn Pipeline with preprocessing + regressor, saved as `best_model.pkl`.")


def page_predict():
    hero("Make a Prediction", "Enter block-group details — engineered features are calculated automatically")
    model, feature_info = load_model_artifacts()
    feature_names = feature_info["features"]

    info_card(
        "How it works",
        "You enter <strong>9 raw inputs</strong> below. The app then computes "
        "<code>houseAgeLabel</code> (from housing age) and <code>valuePerRoom</code> "
        "(from income and rooms) before calling the model — same as in training.",
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            longitude = st.number_input("Longitude", value=-118.0, format="%.4f")
            latitude = st.number_input("Latitude", value=34.0, format="%.4f")
            housing_median_age = st.number_input(
                "Housing Median Age (years)", value=28, min_value=1, step=1, format="%d",
            )
            total_rooms = st.number_input(
                "Total Rooms", value=2000, min_value=1, step=1, format="%d",
            )
            total_bedrooms = st.number_input(
                "Total Bedrooms", value=400, min_value=1, step=1, format="%d",
            )
        with col2:
            population = st.number_input(
                "Population", value=1200, min_value=1, step=1, format="%d",
            )
            households = st.number_input(
                "Households", value=400, min_value=1, step=1, format="%d",
            )
            median_income = st.number_input(
                "Median Income (×$10k)", value=3.5, min_value=0.1, format="%.2f",
            )
            ocean_proximity = st.selectbox("Ocean Proximity", OCEAN_OPTIONS)

        submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "longitude": float(longitude),
            "latitude": float(latitude),
            "housing_median_age": int(housing_median_age),
            "total_rooms": int(total_rooms),
            "total_bedrooms": int(total_bedrooms),
            "population": int(population),
            "households": int(households),
            "median_income": float(median_income),
            "ocean_proximity": ocean_proximity,
        }])
        featured = engineer_features(input_df)
        prediction = float(model.predict(featured[feature_names])[0])

        st.markdown("### Prediction Result")
        st.markdown(
            f'<div class="result-box"><div>Predicted Median House Value</div>'
            f'<div class="result-price">${prediction:,.0f}</div></div>',
            unsafe_allow_html=True,
        )

        age_label = featured["houseAgeLabel"].iloc[0]
        vpr = featured["valuePerRoom"].iloc[0]
        info_card(
            "Auto-engineered features",
            f"<code>houseAgeLabel</code> = <strong>{age_label}</strong> "
            f"(from age {int(housing_median_age)} years) · "
            f"<code>valuePerRoom</code> = <strong>{vpr:.4f}</strong> "
            f"(from income and rooms per household)",
        )

        st.session_state.prediction_history.insert(0, {
            "Predicted Price": f"${prediction:,.0f}",
            "Median Income": median_income,
            "Ocean Proximity": ocean_proximity,
            "Housing Age": int(housing_median_age),
            "Age Label": age_label,
            "Value/Room": f"{vpr:.4f}",
            "Latitude": latitude,
            "Longitude": longitude,
        })

    if st.session_state.prediction_history:
        st.markdown("### Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)
        if st.button("Clear history"):
            st.session_state.prediction_history = []
            st.rerun()


def page_model_comparison():
    hero("Model Comparison", "7 regression models on the same 80/20 test split")

    metrics = load_metrics()
    if metrics.empty:
        st.warning("Run `python -m src.evaluate` to generate metrics.")
        return

    _, feature_info = load_model_artifacts()
    best_name = feature_info.get("model_name")
    leaderboard = compute_leaderboard(metrics)
    winner = leaderboard.iloc[0]

    st.markdown("### Best Model Selection")
    info_card(
        "Weighted composite score",
        "Composite = 0.40×RMSE rank + 0.35×MAE rank + 0.15×R² rank + 0.10×train-time rank "
        "(rank 1 = best per metric). <strong>Lowest composite wins.</strong> "
        "We use this instead of R² alone because RMSE and MAE reflect real dollar error.",
    )

    table = leaderboard.rename(columns={
        "rank": "Rank",
        "model": "Model",
        "composite_score": "Composite Score",
        "rmse": "RMSE ($)",
        "mae": "MAE ($)",
        "r2": "R²",
        "train_time_sec": "Train (s)",
        "predict_time_sec": "Predict (s)",
    })[[
        "Rank", "Model", "Composite Score", "RMSE ($)", "MAE ($)", "R²", "Train (s)", "Predict (s)",
    ]]

    st.dataframe(
        table.style.format({
            "Composite Score": "{:.3f}",
            "RMSE ($)": "${:,.0f}",
            "MAE ($)": "${:,.0f}",
            "R²": "{:.4f}",
            "Train (s)": "{:.3f}",
            "Predict (s)": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    chart_l, chart_r = st.columns(2)
    with chart_l:
        show_chart(EVAL_DIR / "model_comparison_rmse.png", show_interpretation=False)
        st.caption("RMSE in dollars — lower is better. Tree ensembles lead; SVR fails to generalize.")
    with chart_r:
        show_chart(EVAL_DIR / "model_leaderboard.png", show_interpretation=False)
        st.caption("Composite ranking — RMSE 40%, MAE 35%, R² 15%, train time 10%.")

    info_card(
        f"Winner: {best_name}",
        f"Composite score <strong>{winner['composite_score']:.3f}</strong> — "
        f"RMSE ${winner['rmse']:,.0f}, MAE ${winner['mae']:,.0f}, R² {winner['r2']:.4f}. "
        f"Deployed as <code>best_model.pkl</code> for live predictions.",
    )


def page_eda():
    hero("Exploratory Data Analysis", "Visual patterns that guided feature engineering and modeling")
    eda_files = [
        "house_value_distribution.png",
        "house_age_distribution.png",
        "correlation_heatmap.png",
        "geographical_scatter.png",
        "income_vs_value.png",
        "ocean_proximity_distribution.png",
        "avg_value_by_ocean_proximity.png",
        "pairplot.png",
    ]
    show_all_charts([FIGURES_DIR / f for f in eda_files])


def page_model_analysis():
    hero("Model Analysis", "Feature importance, learning curves, residuals, and cross-validation")
    _, feature_info = load_model_artifacts()
    best_name = feature_info.get("model_name", "Random Forest")
    safe = best_name.lower().replace(" ", "_")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Importance",
        "Learning Curves",
        "Residuals & Predictions",
        "Cross Validation",
    ])

    with tab1:
        st.markdown("##### Best model first, then all other models")
        fi_paths = [EVAL_DIR / f"feature_importance_{safe}.png"]
        fi_paths += sorted(
            p for p in EVAL_DIR.glob("feature_importance_*.png")
            if p.name != f"feature_importance_{safe}.png"
        )
        coef_paths = sorted(EVAL_DIR.glob("coefficients_*.png"))
        show_all_charts(fi_paths + coef_paths)

    with tab2:
        show_all_charts(sorted(EVAL_DIR.glob("learning_curve_*.png")))

    with tab3:
        residual_paths = sorted(EVAL_DIR.glob("residuals_*.png"))
        pred_paths = sorted(EVAL_DIR.glob("pred_vs_actual_*.png"))
        for res_path, pred_path in zip(residual_paths, pred_paths):
            show_chart(res_path)
            show_chart(pred_path)

    with tab4:
        cv = load_cv_results()
        if not cv:
            st.warning("Cross-validation file not found. Run `python -m src.evaluate`.")
        else:
            rows = []
            for name, scores in cv.items():
                rows.append({
                    "Model": name,
                    "Mean R²": scores["mean_score"],
                    "Std R²": scores["std_score"],
                    "Folds": scores["n_splits"],
                })
            cv_df = pd.DataFrame(rows).sort_values("Mean R²", ascending=False)
            st.dataframe(
                cv_df.style.format({"Mean R²": "{:.4f}", "Std R²": "{:.4f}"}),
                use_container_width=True,
            )
            best_cv = cv_df.iloc[0]
            worst_cv = cv_df.iloc[-1]
            info_card(
                "Interpretation",
                f"5-fold CV on the training set (not the hold-out test). "
                f"<strong>{best_cv['Model']}</strong> leads with mean R² "
                f"<strong>{best_cv['Mean R²']:.3f} ± {best_cv['Std R²']:.3f}</strong> — "
                f"highest score and lowest variance. "
                f"<strong>Gradient Boosting</strong> is second (~0.784). Linear models cluster at ~0.649. "
                f"<strong>{worst_cv['Model']}</strong> is negative on every fold "
                f"({worst_cv['Mean R²']:.3f}), confirming it does not generalize. "
                f"CV aligns with the 80/20 test winner (Random Forest test R² 0.806).",
            )


def page_about():
    hero("About the Team", "IDAE Section 4 — Group 2")
    about_section(
        "Team Members",
        "<ul>"
        "<li>Eleni Andualem</li>"
        "<li>Elias Berhanu</li>"
        "<li>Selam Elias</li>"
        "</ul>",
    )
    about_section(
        "Tech Stack",
        "<p>Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib · seaborn · joblib · pytest</p>",
    )
    about_section(
        "What This App Does",
        "<p>This web application lets anyone enter property details — location, income, rooms, ocean proximity — "
        "and get an instant median house price prediction from our best model (Random Forest). It also presents "
        "the full EDA, model comparison, feature importance, learning curves, residual analysis, and cross-validation "
        "results from our end-to-end ML pipeline.</p>",
    )
    about_section(
        "What We Built & Learned",
        "<p>Through this project we learned how to take a raw dataset from exploration to a deployed product: "
        "cleaning missing values, engineering meaningful features, training and comparing seven regression models, "
        "selecting the best one with a weighted composite score, and packaging everything into an interactive Streamlit app. "
        "We gained hands-on experience with sklearn Pipelines, model evaluation metrics, visualization, and making "
        "ML results understandable to non-technical users through chart interpretations.</p>",
    )
    about_section(
        "Project Completed",
        "<p><strong>August 2, 2026</strong></p>",
    )


def main():
    st.set_page_config(
        page_title="CA Housing Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_session_state()

    st.sidebar.markdown("## 🏠 CA Housing")
    st.sidebar.caption("Median price prediction app")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Predict",
            "Model Comparison",
            "EDA",
            "Model Analysis",
            "About Team",
        ],
    )

    pages = {
        "Home": page_home,
        "Predict": page_predict,
        "Model Comparison": page_model_comparison,
        "EDA": page_eda,
        "Model Analysis": page_model_analysis,
        "About Team": page_about,
    }
    pages[page]()

    st.sidebar.markdown("---")
    st.sidebar.caption("California Housing ML · Group 2")


if __name__ == "__main__":
    main()
