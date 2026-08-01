# Clean raw housing data

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import DEFAULT_DATA_PATH, load_data

TARGET_COLUMN = "median_house_value"
NUMERIC_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    TARGET_COLUMN,
]
CATEGORICAL_COLUMNS = ["ocean_proximity"]
BINARY_COLUMNS: list[str] = []


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate rows
    return df.drop_duplicates().reset_index(drop=True)


def trim_and_normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    # Clean text columns
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )
        cleaned[col] = cleaned[col].replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA})
    return cleaned


def parse_numeric_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Convert columns to numbers
    cleaned = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(
                cleaned[col].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )
    return cleaned


def convert_binary_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Map yes/no values to 0 and 1
    cleaned = df.copy()
    binary_map = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }
    for col in BINARY_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(binary_map)
            )
    return cleaned

def build_preprocessor(numeric_features : list[str], categorical_features : list[str], scale_numeric : bool = True) -> ColumnTransformer:
    # Build sklearn ColumnTransformer with imputation, scaling, and one-hot encoding
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps = numeric_steps)
    
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with bad values
    cleaned = df.copy()
    valid = pd.Series(True, index=cleaned.index)

    if TARGET_COLUMN in cleaned.columns:
        valid &= cleaned[TARGET_COLUMN].gt(0)

    for col in ["total_rooms", "total_bedrooms", "population", "households", "housing_median_age"]:
        if col in cleaned.columns:
            valid &= cleaned[col].isna() | cleaned[col].ge(0)

    if "median_income" in cleaned.columns:
        valid &= cleaned["median_income"].gt(0)

    if {"total_bedrooms", "total_rooms"}.issubset(cleaned.columns):
        valid &= cleaned["total_bedrooms"].isna() | cleaned["total_bedrooms"].le(cleaned["total_rooms"])

    if {"longitude", "latitude"}.issubset(cleaned.columns):
        valid &= cleaned["longitude"].between(-125, -114)
        valid &= cleaned["latitude"].between(32, 42)

    return cleaned.loc[valid].reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing numbers with median, text with 'unknown'
    cleaned = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in CATEGORICAL_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("unknown")
    return cleaned


def create_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Run all cleaning steps
    cleaned = remove_duplicates(df)
    cleaned = trim_and_normalize_text(cleaned)
    cleaned = parse_numeric_fields(cleaned)
    cleaned = convert_binary_fields(cleaned)
    cleaned = remove_invalid_values(cleaned)
    cleaned = handle_missing_values(cleaned)
    return cleaned


if __name__ == "__main__":
    raw_df = load_data()
    clean_df = create_clean_dataframe(raw_df)
    print(f"Raw shape: {raw_df.shape}")
    print(f"Clean shape: {clean_df.shape}")
    print(f"Missing values: {clean_df.isnull().sum().sum()}")
