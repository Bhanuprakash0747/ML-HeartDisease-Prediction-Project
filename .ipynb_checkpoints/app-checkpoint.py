import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease ML App",
    page_icon="❤️",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("BITS ML ASSIGNEMNT 2 - ❤️ Heart Disease Prediction Dashboard")
st.markdown("### End-to-End Machine Learning Project by 2025AA05782@wilp.bits-pilani.ac.in")

st.markdown(
"""
This app predicts **heart disease risk** using multiple ML models.
Upload a dataset to evaluate model performance.
"""
)

# ---------------- SIDEBAR ----------------
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

# ---------------- MODEL DESCRIPTIONS ----------------
model_info = {
    "Logistic Regression":"Baseline linear model for binary classification.",
    "Decision Tree":"Tree-based model, easy to interpret.",
    "KNN":"Distance-based model using nearest neighbors.",
    "Naive Bayes":"Probabilistic model assuming independence.",
    "Random Forest":"Ensemble of trees, robust and accurate.",
    "XGBoost":"Advanced boosting algorithm with high performance."
}

st.sidebar.info(model_info[model_name])

# ---------------- MAIN ----------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Load model
    model = joblib.load(f"model/{model_name}.pkl")

    X = df.drop("target", axis=1)
    y = df["target"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    # ---------------- METRICS ----------------
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

    # ---------------- CONFUSION MATRIX HEATMAP ----------------
    st.subheader("🔥 Confusion Matrix")

    cm = confusion_matrix(y,y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease","Disease"],
                yticklabels=["No Disease","Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)

    # ---------------- REPORT ----------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y,y_pred))

    # ---------------- DOWNLOAD ----------------
    st.subheader("⬇️ Download Predictions")

    output = df.copy()
    output["Prediction"] = y_pred

    csv = output.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )

else:
    st.info("⬅️ Upload a CSV file from the sidebar to begin")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
"Built with ❤️ using Streamlit & Scikit-learn | ML Portfolio Project"
)
