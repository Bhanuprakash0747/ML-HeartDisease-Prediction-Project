# ❤️ Heart Disease Prediction ML Project

## Problem Statement
The goal of this project is to predict whether a patient has heart disease based on medical attributes such as age, cholesterol level, heart rate, and other clinical features.

This is a binary classification problem where:
- 0 = No Heart Disease
- 1 = Heart Disease

---

## Dataset Description
The Heart Disease dataset contains medical and diagnostic information for patients.

Features include:
- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Rest ECG (restecg)
- Maximum Heart Rate (thalachh)
- Exercise-induced angina (exang)
- Oldpeak
- Slope
- Number of major vessels (ca)
- Thalassemia (thal)

Target:
- target (0 or 1)

Dataset size: 1600+ records

---

## Models Used
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

---

## Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|---------|-----|----------|------|----|----|
| Logistic Regression | 0.766 | 0.833 | 0.746 |0.832 | 0.787 | 0.534 |
| Decision Tree |0.990 | 0.990 | 0.991 | 0.990 | 0.990| 0.980 |
| KNN | 0.947 | 0.985 |0.922 | 0.980 | 0.950 | 0.895 |
| Naive Bayes | 0.735 | 0.820 | 0.720 | 0.799 | 0.758 | 0.471 |
| Random Forest | 0.992 | 1.000 | 0.994 | 0.991 | 0.991 | 0.984 |
| XGBoost | 0.990 | 0.998 | 0.990 | 0.989 | 0.992 | 0.980 |

---

## Observations
| ML Model Name | Observation about model performance|
|------|---------|
| Logistic Regression | Provided a good baseline not performed best |
| Decision Tree | Performed well but may overfit with dataset | 
| KNN |KNN showed Strong recall when compared to other models recall |
| Naive Bayes | The lowest performance due to feature independence assumptions with datasets | 
| Random Forest | Achieved the best performance across all metrics best model for this dataset |
| XGBoost | Performed strongly and was a close second |

