import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.feature_engineering import engineer_features, list_engineered_features
from src.predict import load_feature_info, load_model, predict_price
from src.preprocessing import create_clean_dataframe
from src.utils import DEFAULT_DATA_PATH, MODELS_DIR, load_data


@pytest.fixture
def sample_row() -> dict:
    return {
        "longitude": -118.0,
        "latitude": 34.0,
        "housing_median_age": 28.0,
        "total_rooms": 2000.0,
        "total_bedrooms": 400.0,
        "population": 1200.0,
        "households": 400.0,
        "median_income": 3.5,
        "ocean_proximity": "INLAND",
    }


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert DEFAULT_DATA_PATH.exists()


def test_create_clean_dataframe_has_no_missing_values():
    clean_df = create_clean_dataframe(load_data())
    assert clean_df.isnull().sum().sum() == 0
    assert len(clean_df) > 0


def test_engineer_features_adds_columns(sample_row):
    featured = engineer_features(pd.DataFrame([sample_row]))
    for col in list_engineered_features():
        assert col in featured.columns


def test_model_files_exist():
    assert (MODELS_DIR / "best_model.pkl").exists()
    assert (MODELS_DIR / "feature_info.json").exists()


def test_load_model_and_feature_info():
    model = load_model()
    info = load_feature_info()
    assert hasattr(model, "predict")
    assert "features" in info
    assert len(info["features"]) > 0


def test_predict_price_returns_positive_value(sample_row):
    price = predict_price(sample_row)
    assert isinstance(price, float)
    assert price > 0


def test_comparison_metrics_file_exists():
    assert Path("outputs/metrics/comparison.csv").exists()
