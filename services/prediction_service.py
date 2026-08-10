from services.model_service import ModelService
from utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Handles model inference for heart disease prediction."""

    def __init__(self):
        self.model_service = ModelService()

    def predict(self, X, model_name: str):
        """Generate predictions and probabilities for input data."""

        if X is None:
            logger.error("Prediction input cannot be None.")
            raise ValueError("Prediction input cannot be None.")

        if len(X) == 0:
            logger.warning("Prediction input contains no rows.")
            raise ValueError("Prediction input cannot be empty.")

        logger.info(
            "Starting prediction using model '%s' for %d records.",
            model_name,
            len(X),
        )

        try:
            scaler = self.model_service.load_scaler()
            X_scaled = scaler.transform(X)

            model = self.model_service.load_model(model_name)

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            logger.info(
                "Prediction completed successfully for %d records.",
                len(X),
            )

            return predictions, probabilities

        except Exception as exc:
            logger.error(
                "Prediction failed using model '%s': %s",
                model_name,
                exc,
            )
            raise
