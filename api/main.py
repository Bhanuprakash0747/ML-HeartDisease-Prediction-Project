import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import FEATURE_COLUMNS, PRODUCTION_MODEL
from services.prediction_service import PredictionService
from utils.logger import get_logger


logger = get_logger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST API for heart disease prediction using the production ML model.",
    version="1.0.0",
)


class HeartDiseaseRequest(BaseModel):
    age: float = Field(..., ge=0)
    sex: float = Field(..., ge=0)
    cp: float = Field(..., ge=0)
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: float = Field(..., ge=0)
    restecg: float = Field(..., ge=0)
    thalachh: float = Field(..., ge=0)
    exang: float = Field(..., ge=0)
    oldpeak: float
    slope: float = Field(..., ge=0)
    ca: float = Field(..., ge=0)
    thal: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model: str


prediction_service = PredictionService()


@app.get("/health")
def health_check():
    """Check whether the API is running."""

    return {
        "status": "healthy",
        "model": PRODUCTION_MODEL,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: HeartDiseaseRequest):
    """Generate a heart disease prediction."""

    try:
        input_data = request.model_dump()

        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURE_COLUMNS,
        )

        logger.info(
            "Received prediction request using production model '%s'.",
            PRODUCTION_MODEL,
        )

        predictions, probabilities = prediction_service.predict(
            input_df,
            PRODUCTION_MODEL,
        )

        prediction = int(predictions[0])
        probability = float(probabilities[0])

        logger.info(
            "Prediction completed successfully. Prediction: %d",
            prediction,
        )

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model=PRODUCTION_MODEL,
        )

    except ValueError as exc:
        logger.error("Invalid prediction request: %s", exc)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.error("Prediction API failed: %s", exc)

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error.",
        )