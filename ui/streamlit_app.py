import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from services.prediction_service import PredictionService
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Heart Disease ML App",
                   page_icon="❤️",
                   layout="wide")

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("### End-to-End Machine Learning Project")

st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression","Decision Tree",
     "KNN","Naive Bayes",
     "Random Forest","XGBoost"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

st.sidebar.markdown("### 📥 Download Sample Dataset")

sample_df = pd.read_csv("data/heartdisease_dataset.csv")

sample_csv = sample_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    "Download Sample CSV",
    sample_csv,
    "heartdisease_dataset.csv",
    "text/csv"
)

model_info = {
    "Logistic Regression":"Baseline linear models.",
    "Decision Tree":"Easy to interpret.",
    "KNN":"Nearest neighbor based.",
    "Naive Bayes":"Probabilistic models.",
    "Random Forest":"Robust ensemble models.",
    "XGBoost":"High-performance boosting."
}

st.sidebar.info(model_info[model_name])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    service = PredictionService()

    y_pred, y_prob = service.predict(
        X,
        model_name
    )

    st.subheader("📈 Performance Metrics")

    acc = accuracy_score(y,y_pred)
    prec = precision_score(y,y_pred)
    rec = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    mcc = matthews_corrcoef(y,y_pred)
    auc = roc_auc_score(y,y_prob)

    c1,c2,c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("AUC", f"{auc:.3f}")
    c3.metric("F1 Score", f"{f1:.3f}")

    c4,c5,c6 = st.columns(3)
    c4.metric("Precision", f"{prec:.3f}")
    c5.metric("Recall", f"{rec:.3f}")
    c6.metric("MCC", f"{mcc:.3f}")

    st.subheader("🔥 Confusion Matrix")

    cm = confusion_matrix(y,y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease","Disease"],
                yticklabels=["No Disease","Disease"])
    st.pyplot(fig)

    st.subheader("📄 Classification Report")
    st.text(classification_report(y,y_pred))

    st.subheader("⬇️ Download Predictions")

    df["Prediction"] = y_pred
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )

else:
    st.info("⬅️ Upload a CSV to begin")
