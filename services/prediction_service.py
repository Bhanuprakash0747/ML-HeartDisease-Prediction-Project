from services.model_service import ModelService


class PredictionService:
    """Handles model inference for heart disease prediction."""

    def __init__(self):
        self.model_service = ModelService()

    def predict(self, X, model_name: str):
        """Generate predictions and probabilities for input data."""

        scaler = self.model_service.load_scaler()
        X_scaled = scaler.transform(X)

        model = self.model_service.load_model(model_name)

        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities