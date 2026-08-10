import numpy as np
import pandas as pd

from utils.logger import get_logger


logger = get_logger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the input heart disease dataset.

    Replaces '?' with NaN, converts values to numeric,
    and removes rows containing missing values.
    """

    if df is None:
        logger.error("Input DataFrame is None.")
        raise ValueError("Input DataFrame cannot be None.")

    if df.empty:
        logger.warning("Input DataFrame is empty.")
        raise ValueError("Input DataFrame cannot be empty.")

    logger.info(
        "Starting preprocessing. Input rows: %d, columns: %d",
        df.shape[0],
        df.shape[1],
    )

    processed_df = df.copy()

    missing_markers = (processed_df == "?").sum().sum()

    if missing_markers > 0:
        logger.warning(
            "Found %d '?' missing-value markers.",
            missing_markers,
        )

    processed_df = processed_df.replace("?", np.nan)

    processed_df = processed_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    rows_before_drop = len(processed_df)

    processed_df = processed_df.dropna()

    rows_removed = rows_before_drop - len(processed_df)

    if rows_removed > 0:
        logger.warning(
            "Removed %d rows containing missing or invalid values.",
            rows_removed,
        )

    logger.info(
        "Preprocessing completed. Remaining rows: %d",
        len(processed_df),
    )

    return processed_df