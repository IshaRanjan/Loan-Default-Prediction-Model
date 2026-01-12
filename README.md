# Loan Default Prediction Model

An end-to-end machine learning pipeline for predicting loan default risk using **XGBoost**, with feature reduction, class imbalance handling, and model explainability.

---

## 📌 Project Overview

This project builds a binary classification model to predict whether a loan applicant is likely to default.  
The pipeline focuses on producing a high-performance, interpretable model suitable for real-world financial risk analysis.

Key goals:
- Improve prediction accuracy for default risk
- Reduce noise and redundancy in features
- Handle imbalanced class distribution
- Provide transparent, explainable predictions

---

## 🧠 Methodology

### 1. Feature Engineering & Selection
To reduce overfitting and improve model efficiency:
- Highly correlated features were removed using correlation analysis
- **Random Forest feature importance** was used to identify the most informative variables

### 2. Handling Class Imbalance
Loan default data is typically highly imbalanced.  
To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the dataset before training.

### 3. Model Training
The final classifier was built using **XGBoost**, chosen for its:
- High performance on tabular data
- Built-in regularization
- Ability to handle complex feature interactions

### 4. Hyperparameter Optimization
To maximize performance:
- **Cross-validation** was used
- Hyperparameters were tuned using grid/random search strategies

### 5. Model Explainability
To make predictions transparent and trustworthy:
- **SHAP (SHapley Additive exPlanations)** was used
- Feature contributions were visualized to explain individual predictions and global model behavior

---

## 📊 Model Performance

The trained model achieved strong performance across standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

This makes it suitable for financial risk screening and loan approval workflows.

---

## 🛠 Tech Stack

- Python  
- XGBoost  
- Scikit-learn  
- Pandas, NumPy  
- SMOTE (imblearn)  
- SHAP  
- Matplotlib / Seaborn  

