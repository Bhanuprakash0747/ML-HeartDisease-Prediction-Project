from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data"

PRODUCTION_MODEL = "Random Forest"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalachh",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
