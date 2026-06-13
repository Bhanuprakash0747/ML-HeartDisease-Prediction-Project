import joblib

class PredictionService:

    def load_scaler(self):
        return joblib.load("models/scaler.pkl")

    def load_model(self, model_name):
        return joblib.load(f"models/{model_name}.pkl")

    def predict(self, X, model_name):
        scaler = self.load_scaler()
        X_scaled = scaler.transform(X)
        model = self.load_model(model_name)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:,1]
        return predictions, probabilities