import pandas as pd
import pytest

from utils.preprocessing import preprocess


def test_preprocess_replaces_missing_values():
    df = pd.DataFrame(
        {
            "age": [55, 60],
            "chol": [250, "?"],
        }
    )

    result = preprocess(df)

    assert result.isna().sum().sum() == 0


def test_preprocess_converts_values_to_numeric():
    df = pd.DataFrame(
        {
            "age": ["55", "60"],
            "chol": ["250", "300"],
        }
    )

    result = preprocess(df)

    assert pd.api.types.is_numeric_dtype(result["age"])
    assert pd.api.types.is_numeric_dtype(result["chol"])


def test_preprocess_rejects_none():
    with pytest.raises(ValueError):
        preprocess(None)


def test_preprocess_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        preprocess(pd.DataFrame())
