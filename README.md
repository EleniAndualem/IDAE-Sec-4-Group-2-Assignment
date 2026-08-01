# California Housing Price Prediction

This is an end-to-end Machine Learning project to predict the median house value for California districts based on 1990 census data. 

## Problem Statement

The goal of this project is to build a regression model that accurately predicts the `median_house_value` for a given block group in California. This requires handling missing data, engineering meaningful features (like age categories and value per room), and comparing multiple regression algorithms to find the optimal balance of accuracy and computational efficiency.

## Dataset Description

The project uses the **California Housing Prices** dataset. Each row represents one block group (a district) in California. 

Key features include:
* **Location:** `longitude`, `latitude`, `ocean_proximity`
* **Housing Details:** `housing_median_age`, `total_rooms`, `total_bedrooms`
* **Demographics:** `population`, `households`, `median_income`
* **Target Variable:** `median_house_value`

## Project Structure

```text
├── app/                  # Streamlit application for the interactive UI
├── data/                 # Raw and processed datasets
├── models/               # Saved machine learning models (e.g., best_model.pkl)
├── notebooks/            # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── outputs/              # Generated assets
│   ├── figures/          # Plots for EDA, feature importance, and model evaluation
│   ├── metrics/          # CSV/JSON files with evaluation scores (e.g. cross_validation.json)
│   └── reports/          # Markdown reports (e.g. model_comparison.md)
├── src/                  # Source code for the ML pipeline
│   ├── evaluate.py       # Model evaluation, comparison, and leaderboard generation
│   ├── feature_engineering.py # Logic for deriving new features
│   ├── preprocessing.py  # Data cleaning, imputation, and scaling
│   ├── train.py          # Training orchestrator for all models
│   ├── utils.py          # Utility functions for I/O and setup
│   └── visualization.py  # Code for generating plots
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (this file)
```

## Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd IDAE-Sec-4-Group-2-Assignment
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Commands

Run the following commands from the root of the project (`IDAE-Sec-4-Group-2-Assignment/`).

**1. Train and Evaluate Models:**
To run the entire pipeline (cleaning, feature engineering, training, and evaluation), run:
```bash
python -m src.evaluate
```
*(This will generate the best model in `models/best_model.pkl` and save all evaluation charts to `outputs/figures/evaluation/`)*

**2. Run the Streamlit App:**
To launch the interactive web interface where you can explore the data and make predictions:
```bash
streamlit run app/app.py
```

**3. Run Unit Tests:**
To verify that the project is working properly:
```bash
PYTHONPATH=. pytest tests/ -v
```

## Model Comparison Results

Seven different regression models were trained and evaluated: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and SVR.

The **Random Forest** model achieved the best balance of accuracy and training efficiency, emerging as the top performer on the leaderboard.

| Rank | Model | RMSE | MAE | R² | Train Time (s) | Predict Time (s) |
|------|-------|------|-----|----|----------------|------------------|
| 1 | Random Forest ⭐ | $50,476 | $32,960 | 0.8056 | 0.371 | 0.0162 |
| 2 | Gradient Boosting | $54,410 | $37,051 | 0.7741 | 1.983 | 0.0071 |
| 3 | Decision Tree | $67,437 | $42,981 | 0.6530 | 0.206 | 0.0032 |

*(Full details can be found in `outputs/reports/model_comparison.md`)*

## Outputs & Screenshots

All generated visualizations are automatically saved in the `outputs/figures/` directory, including:
* **EDA Plots:** Pairplots, correlation heatmaps, and price distributions.
* **Evaluation Plots:** Actual vs. Predicted scatter plots, learning curves, and residual plots for every model.
* **Feature Importance:** Bar charts showing which features (e.g., median income) had the highest impact on the predictions.