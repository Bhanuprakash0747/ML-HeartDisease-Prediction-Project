from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "Random Forest"


def test_prediction_endpoint():
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalachh": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert data["model"] == "Random Forest"


def test_prediction_rejects_invalid_age():
    payload = {
        "age": -5,
        "sex": 1,
        "cp": 1,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalachh": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
