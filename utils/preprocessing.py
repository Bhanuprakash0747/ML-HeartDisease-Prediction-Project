import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the input heart disease dataset.

    Replaces '?' with NaN, converts values to numeric,
    and removes rows containing missing values.
    """

    processed_df = df.copy()

    processed_df = processed_df.replace("?", np.nan)

    processed_df = processed_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    processed_df = processed_df.dropna()

    return processed_df