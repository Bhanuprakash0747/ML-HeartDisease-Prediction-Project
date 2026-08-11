import pandas as pd

from utils.quality_metrics import (
    calculate_missing_value_rate,
    calculate_model_quality,
    check_schema,
)


def test_schema_is_valid():
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalachh",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    df = pd.DataFrame(columns=columns)

    assert check_schema(df) is True


def test_schema_is_invalid_when_column_missing():
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalachh",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "target",
    ]

    df = pd.DataFrame(columns=columns)

    assert check_schema(df) is False


def test_missing_value_rate():
    df = pd.DataFrame(
        {
            "age": [55, "?"],
            "sex": [1, 0],
            "target": [1, 0],
        }
    )

    missing_rate = calculate_missing_value_rate(df)

    assert missing_rate == (1 / 6) * 100


def test_model_quality_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.4]

    metrics = calculate_model_quality(
        y_true,
        y_pred,
        y_prob,
    )

    assert "f1_score" in metrics
    assert "roc_auc" in metrics

    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
