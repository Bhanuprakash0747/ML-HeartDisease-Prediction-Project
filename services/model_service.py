import joblib

from config import MODEL_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """Handles loading trained ML models and preprocessing artifacts."""

    def load_scaler(self):
        """Load the trained feature scaler."""

        scaler_path = MODEL_PATH / "scaler.pkl"

        logger.info("Loading scaler from: %s", scaler_path)

        try:
            scaler = joblib.load(scaler_path)

            logger.info("Scaler loaded successfully.")

            return scaler

        except FileNotFoundError:
            logger.error(
                "Scaler file not found: %s",
                scaler_path,
            )
            raise

        except Exception as exc:
            logger.error(
                "Failed to load scaler: %s",
                exc,
            )
            raise

    def load_model(self, model_name: str):
        """Load a trained model by its name."""

        if not model_name:
            logger.error("Model name was not provided.")
            raise ValueError("Model name cannot be empty.")

        model_path = MODEL_PATH / f"{model_name}.pkl"

        logger.info("Loading model: %s", model_name)

        try:
            model = joblib.load(model_path)

            logger.info(
                "Model '%s' loaded successfully.",
                model_name,
            )

            return model

        except FileNotFoundError:
            logger.error(
                "Model file not found: %s",
                model_path,
            )
            raise

        except Exception as exc:
            logger.error(
                "Failed to load model '%s': %s",
                model_name,
                exc,
            )
            raise
