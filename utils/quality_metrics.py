import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from config import FEATURE_COLUMNS

EXPECTED_COLUMNS = FEATURE_COLUMNS + ["target"]


def check_schema(df: pd.DataFrame) -> bool:
    """Check whether the dataset contains the expected columns."""

    return list(df.columns) == EXPECTED_COLUMNS


def calculate_missing_value_rate(df: pd.DataFrame) -> float:
    """Calculate the percentage of missing values in the raw dataset."""

    if df.empty:
        raise ValueError("Cannot calculate data quality for an empty DataFrame.")

    cleaned_df = df.replace("?", pd.NA)

    missing_values = cleaned_df.isna().sum().sum()
    total_values = cleaned_df.shape[0] * cleaned_df.shape[1]

    return (missing_values / total_values) * 100


def calculate_model_quality(
    y_true,
    y_pred,
    y_prob,
) -> dict:
    """Calculate model-quality metrics."""

    return {
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
