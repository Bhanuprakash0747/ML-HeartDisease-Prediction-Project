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
| Logistic Regression | 0.7249 | 0.8016 | 0.6903 | 0.8211 | 0.7500 | 0.4576 |
| Decision Tree | 0.9524 | 0.9524 | 0.9526 | 0.9526 | 0.9526 | 0.9048 |
| KNN | 0.9021 | 0.9489 | 0.8525 | 0.9737 | 0.9091 | 0.8124 |
| Naive Bayes | 0.6878 | 0.7761 | 0.6607 | 0.7789 | 0.7150 | 0.3813 |
| Random Forest | 0.9603 | 0.9942 | 0.9630 | 0.9579 | 0.9604 | 0.9206 |
| XGBoost | 0.9497 | 0.9865 | 0.9430 | 0.9579 | 0.9504 | 0.8996 |

---

## Observations

Logistic Regression - Logistic Regression provided a good baseline not performed best.
Decision Tree - Decision Tree performed well but may overfit with dataset.
KNN - KNN showed strong recall when compared to other models recall.
Navie Bayes - Naive Bayes had the lowest performance due to feature independence assumptions with datasets.
Random Forest - Random Forest achieved the best performance across all metrics best model for this dataset.
XGBoost - XGBoost also performed strongly and was a close second.

