# Customer Churn Prediction Using Machine Learning

## Overview

Customer churn prediction is a critical problem for subscription-based businesses, particularly in the telecommunications industry. This project develops and evaluates multiple machine learning models to predict whether a customer is likely to discontinue a service.

The project follows the CRISP-DM methodology and compares four classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

In addition to predictive modelling, the project incorporates:

- Statistical feature validation using Statsmodels
- Threshold tuning for Logistic Regression
- SHAP (SHapley Additive Explanations) for model interpretability
- Feature importance analysis
- Overfitting assessment through train-test comparison

---

## Objectives

1. Predict customer churn using machine learning models.
2. Identify the most influential factors contributing to customer churn.
3. Compare model performance under class imbalance conditions.
4. Provide explainable AI insights through SHAP analysis.

---

## Dataset

The project uses a telecom customer churn dataset containing customer demographic, billing, and subscription information.

### Features

| Feature | Type |
|----------|----------|
| Age | Numerical |
| Gender | Categorical |
| Tenure | Numerical |
| MonthlyCharges | Numerical |
| TotalCharges | Numerical |
| Contract | Categorical |
| PaymentMethod | Categorical |
| Churn | Target Variable |

---

## Project Workflow

### 1. Data Preprocessing

- Handling missing values
- Label encoding categorical features
- Feature scaling using StandardScaler
- Stratified train-test split (80:20)

### 2. Model Development

Implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### 3. Evaluation Metrics

Primary metric:

- F1 Score

Additional metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix
- ROC Analysis

### 4. Explainability

- SHAP TreeExplainer
- Feature Importance Analysis
- Statistical Validation (AIC & p-values)

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 0.677 | 0.509 | 0.716 | 0.595 |
| Decision Tree | 0.753 | 0.630 | 0.620 | **0.625** |
| Random Forest | 0.731 | 0.621 | 0.485 | 0.544 |
| XGBoost | 0.725 | 0.574 | 0.661 | 0.614 |

### Key Findings

- Decision Tree achieved the highest F1 Score (0.625).
- Random Forest showed significant overfitting.
- Logistic Regression provided the best statistical interpretability.
- XGBoost delivered strong predictive performance with explainable outputs.

---

## Feature Importance Findings

The most influential churn predictors were:

1. Contract Type
2. Monthly Charges
3. Tenure

These features were consistently identified across:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SHAP Analysis

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Statsmodels
- SHAP
- Matplotlib
- Seaborn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git

cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the notebook or Python scripts:

```bash
jupyter notebook
```

or

```bash
python churn_prediction.py
```

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
├── notebooks/
├── models/
├── outputs/
│   ├── confusion_matrices/
│   ├── shap_plots/
│   └── feature_importance/
│
├── Customer_Churn_Report.pdf
├── requirements.txt
├── README.md
└── churn_prediction.py
```

---

## Future Improvements

- Hyperparameter optimization
- Cross-validation-based threshold tuning
- Deep learning approaches (TabNet)
- Real-time churn prediction pipelines
- Counterfactual explanations and LIME
- Survival analysis for time-to-churn prediction

---

## Authors

- Kavya Kumar
- Modi Eyobo
- Sumukha Sagar

School of Computing  
Dublin City University

---

## License

This project is released under the MIT License.
