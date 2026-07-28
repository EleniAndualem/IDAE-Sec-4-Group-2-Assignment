# Create new features for modeling

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    # Safe division (no divide-by-zero)
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add houseAgeLabel and valuePerRoom
    featured = df.copy()

    featured["houseAgeLabel"] = pd.cut(
        featured["housing_median_age"],
        bins=[0, 15, 30, 45, 60],
        labels=["new", "mid", "mature", "old"],
        include_lowest=True,
    ).astype(str)

    rooms_per_household = _safe_divide(featured["total_rooms"], featured["households"])
    featured["valuePerRoom"] = _safe_divide(featured["median_income"], rooms_per_household)
    featured["valuePerRoom"] = featured["valuePerRoom"].fillna(featured["valuePerRoom"].median())

    return featured


def list_engineered_features() -> list[str]:
    # List new feature names
    return ["houseAgeLabel", "valuePerRoom"]


if __name__ == "__main__":
    from src.preprocessing import create_clean_dataframe
    from src.utils import load_data

    clean_df = create_clean_dataframe(load_data())
    featured_df = engineer_features(clean_df)

    print(f"Original columns: {len(clean_df.columns)}")
    print(f"Featured columns: {len(featured_df.columns)}")
    print(featured_df[list_engineered_features()].head())
