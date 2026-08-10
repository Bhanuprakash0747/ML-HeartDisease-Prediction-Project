import joblib

from config import MODEL_PATH


class ModelService:
    """Handles loading trained ML models and preprocessing artifacts."""

    def load_scaler(self):
        """Load the trained feature scaler."""
        scaler_path = MODEL_PATH / "scaler.pkl"
        return joblib.load(scaler_path)

    def load_model(self, model_name: str):
        """Load a trained model by its name."""
        model_path = MODEL_PATH / f"{model_name}.pkl"
        return joblib.load(model_path)