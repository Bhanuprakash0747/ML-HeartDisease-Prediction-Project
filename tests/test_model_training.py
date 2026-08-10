import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_random_forest_can_fit_small_batch():
    X_small = np.array(
        [
            [55, 1, 1, 140, 250, 0, 1, 150, 0, 1.2, 2, 0, 2],
            [35, 0, 2, 120, 200, 0, 0, 170, 0, 0.5, 1, 0, 2],
            [65, 1, 3, 160, 280, 1, 1, 130, 1, 2.0, 2, 1, 3],
            [45, 0, 0, 110, 180, 0, 0, 175, 0, 0.2, 1, 0, 2],
        ]
    )

    y_small = np.array([1, 0, 1, 0])

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X_small, y_small)

    predictions = model.predict(X_small)

    assert len(predictions) == len(y_small)
    assert set(predictions).issubset({0, 1})


def test_random_forest_can_overfit_small_batch():
    X_small = np.array(
        [
            [55, 1, 1, 140, 250, 0, 1, 150, 0, 1.2, 2, 0, 2],
            [35, 0, 2, 120, 200, 0, 0, 170, 0, 0.5, 1, 0, 2],
            [65, 1, 3, 160, 280, 1, 1, 130, 1, 2.0, 2, 1, 3],
            [45, 0, 0, 110, 180, 0, 0, 175, 0, 0.2, 1, 0, 2],
        ]
    )

    y_small = np.array([1, 0, 1, 0])

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )

    model.fit(X_small, y_small)

    predictions = model.predict(X_small)

    training_accuracy = np.mean(predictions == y_small)

    assert training_accuracy >= 0.75
