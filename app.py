import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

st.title("❤️ Heart Disease Prediction App")

st.write("Upload a heart disease dataset CSV for prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Model selection
    model_name = st.selectbox(
        "Choose Model",
        ["Logistic Regression","Decision Tree",
         "KNN","Naive Bayes",
         "Random Forest","XGBoost"]
    )

    # Load model
    model = joblib.load(f"model/{model_name}.pkl")

    # Split X and y
    X = df.drop("target", axis=1)
    y = df["target"]

    # Prediction
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    st.subheader("📊 Evaluation Metrics")

    acc = accuracy_score(y,y_pred)
    prec = precision_score(y,y_pred)
    rec = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    mcc = matthews_corrcoef(y,y_pred)
    auc = roc_auc_score(y,y_prob)

    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"AUC: {auc:.4f}")
    st.write(f"Precision: {prec:.4f}")
    st.write(f"Recall: {rec:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC: {mcc:.4f}")

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y,y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y,y_pred))
