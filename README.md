# California Housing Price Prediction

**Group 2** · IDAE Section 4 · August 2026

End-to-end regression pipeline that predicts median house values for California census block groups — from data cleaning and feature engineering through model comparison, evaluation, and a deployed web application.

| | |
|---|---|
| **Live App** | [california-housing-price-predictor-group2.streamlit.app](https://california-housing-price-predictor-group2.streamlit.app/) |
| **Technical Report** | [Technical_Report_Group2.pdf](https://drive.google.com/file/d/1Ti8QqplCqqDtdNZijvrSNjBPO6Glw4gs/view?usp=sharing) |
| **Best Model** | Random Forest — RMSE $50,476 · MAE $32,960 · R² 0.806 |

**Team:** Eleni Andualem · Elias Berhanu · Selam Elias

---

## Overview

This project builds a supervised regression system that estimates `median_house_value` (USD) for a California block group from location, housing stock, and demographic features. Seven algorithms are trained on the same 80/20 split and ranked with a weighted composite score (RMSE 40%, MAE 35%, R² 15%, training time 10%). The winning model is served through an interactive Streamlit app with chart interpretations, model diagnostics, and live predictions.

## Dataset

**Source:** `data/California Housing Prices.csv` (1990 US Census)

| | |
|---|---|
| Records | 20,640 block groups |
| Features | 9 inputs + 1 target |
| Missing values | 207 in `total_bedrooms` (imputed) |

| Category | Columns |
|----------|---------|
| Location | `longitude`, `latitude`, `ocean_proximity` |
| Housing | `housing_median_age`, `total_rooms`, `total_bedrooms` |
| Demographics | `population`, `households`, `median_income` |
| Target | `median_house_value` |

**Engineered features:** `houseAgeLabel`, `valuePerRoom`

## Model Results

| Rank | Model | RMSE | MAE | R² |
|------|-------|------|-----|-----|
| 1 | **Random Forest** | $50,476 | $32,960 | 0.806 |
| 2 | Gradient Boosting | $54,380 | $37,041 | 0.774 |
| 3 | Decision Tree | $67,437 | $42,981 | 0.653 |
| 4 | Linear Regression | $69,944 | $50,498 | 0.627 |
| 5 | Lasso Regression | $69,944 | $50,498 | 0.627 |
| 6 | Ridge Regression | $69,951 | $50,504 | 0.627 |
| 7 | SVR | $116,845 | $86,969 | −0.042 |

Full metrics: `outputs/metrics/comparison.csv` · Report: `outputs/reports/model_comparison.md`

## Project Structure

```text
├── app/app.py              # Streamlit web application
├── data/                   # Raw dataset
├── models/                 # best_model.pkl, feature_info.json
├── notebooks/eda.ipynb     # Interactive EDA
├── outputs/
│   ├── figures/            # EDA and evaluation plots
│   ├── metrics/            # comparison.csv, cross_validation.json
│   └── reports/            # Generated reports
├── src/
│   ├── utils.py            # Data loading and inspection
│   ├── preprocessing.py    # Cleaning and preprocessing pipeline
│   ├── feature_engineering.py
│   ├── train.py            # Train all 7 models
│   ├── evaluate.py         # Metrics, CV, model selection
│   ├── predict.py          # CLI prediction
│   └── visualization.py    # Plot generation
├── tests/                  # Unit tests (pytest)
├── Dockerfile              # Container deployment
├── requirements.txt        # Production dependencies
└── requirements-dev.txt    # Dev, test, and SHAP dependencies
```

## Quick Start

```bash
git clone git@github.com:EleniAndualem/IDAE-Sec-4-Group-2-Assignment.git
cd IDAE-Sec-4-Group-2-Assignment
python -m venv .venv
```

**Windows**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: tests, notebooks, SHAP
$env:PYTHONPATH="."
streamlit run app/app.py
```

**Linux / macOS**
```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional
PYTHONPATH=. streamlit run app/app.py
```

## Pipeline Commands

Run from the project root:

```bash
PYTHONPATH=. python -m src.train       # Train all 7 models
PYTHONPATH=. python -m src.evaluate     # Evaluate, compare, save best model
PYTHONPATH=. python -m src.predict      # Single CLI prediction
PYTHONPATH=. pytest tests/ -v         # Run unit tests
```

## Deployment

**Live (Streamlit Cloud)** — already deployed:

[https://california-housing-price-predictor-group2.streamlit.app/](https://california-housing-price-predictor-group2.streamlit.app/)

To redeploy: push to `main` on GitHub. Streamlit Cloud rebuilds automatically from `app/app.py`.

**Docker**

```bash
docker build -t ca-housing-app .
docker run -p 8501:8501 ca-housing-app
```

Open [http://localhost:8501](http://localhost:8501).

## Web Application

Six pages: **Home** · **Predict** · **Model Comparison** · **EDA** · **Model Analysis** · **About Team**

- Live price prediction with engineered features
- Composite model leaderboard with weighting formula
- EDA and evaluation charts with written interpretations
- Feature importance, learning curves, residuals, and cross-validation
- SHAP explainability on predictions (requires `requirements-dev.txt`)

## Documentation

- [Technical Report (PDF)](https://drive.google.com/file/d/1Ti8QqplCqqDtdNZijvrSNjBPO6Glw4gs/view?usp=sharing)
- `outputs/reports/data_understanding.md`
- `outputs/reports/model_comparison.md`

## Tech Stack

Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib · seaborn · joblib · pytest · Docker
