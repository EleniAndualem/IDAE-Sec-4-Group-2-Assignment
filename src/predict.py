# Load saved model and make predictions

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_engineering import engineer_features
from src.utils import MODELS_DIR

MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.json"


def load_model() -> Any:
    # Load trained pipeline from disk
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run: python -m src.evaluate")
    return joblib.load(MODEL_PATH)


def load_feature_info() -> dict:
    # Load feature metadata
    if not FEATURE_INFO_PATH.exists():
        raise FileNotFoundError(f"Feature info not found: {FEATURE_INFO_PATH}")
    return json.loads(FEATURE_INFO_PATH.read_text(encoding="utf-8"))


def predict_price(input_data: dict | pd.DataFrame) -> float:
    # Predict median house value from input features
    model = load_model()
    feature_info = load_feature_info()
    feature_names = feature_info["features"]

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    featured = engineer_features(input_df)
    prediction = model.predict(featured[feature_names])
    return float(prediction[0])


if __name__ == "__main__":
    sample = {
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
    price = predict_price(sample)
    print(f"Predicted median house value: ${price:,.0f}")
