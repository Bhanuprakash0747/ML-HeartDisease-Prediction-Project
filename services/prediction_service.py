import joblib

class PredictionService:

    def load_scaler(self):
        logger.info("Loading scaler")
        return joblib.load("models/scaler.pkl")

    def load_model(self, model_name):
        logger.info(f"Loading model: {model_name}")
        return joblib.load(f"models/{model_name}.pkl")

    def predict(self, X, model_name):

        logger.info("Prediction request received")

        scaler = self.load_scaler()

        logger.info("Applying feature scaling")
        X_scaled = scaler.transform(X)

        model = self.load_model(model_name)

        logger.info("Generating predictions")
        predictions = model.predict(X_scaled)

        logger.info("Generating probabilities")
        probabilities = model.predict_proba(X_scaled)[:,1]

        logger.info("Prediction completed successfully")

        return predictions, probabilities