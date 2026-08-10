import pandas as pd

from config import FEATURE_COLUMNS, PRODUCTION_MODEL
from services.prediction_service import PredictionService


def test_prediction_output_shape():
    service = PredictionService()

    input_data = pd.DataFrame(
        [
            [
                55,
                1,
                1,
                140,
                250,
                0,
                1,
                150,
                0,
                1.2,
                2,
                0,
                2,
            ]
        ],
        columns=FEATURE_COLUMNS,
    )

    predictions, probabilities = service.predict(
        input_data,
        PRODUCTION_MODEL,
    )

    assert len(predictions) == 1
    assert len(probabilities) == 1


def test_prediction_output_range():
    service = PredictionService()

    input_data = pd.DataFrame(
        [
            [
                55,
                1,
                1,
                140,
                250,
                0,
                1,
                150,
                0,
                1.2,
                2,
                0,
                2,
            ]
        ],
        columns=FEATURE_COLUMNS,
    )

    predictions, probabilities = service.predict(
        input_data,
        PRODUCTION_MODEL,
    )

    assert predictions[0] in [0, 1]
    assert 0 <= probabilities[0] <= 1
