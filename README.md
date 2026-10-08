# Customer Churn Prediction

**Which customers are about to leave, and what is pushing them out?**

For a telecom company, losing a customer costs far more than keeping one. This project builds and compares four machine learning models that predict whether a customer will cancel their service, and then looks inside the models to see *why* customers leave.

We followed the CRISP-DM process from data preparation through to evaluation, and put as much effort into explaining the models as into scoring them.

> Group project at Dublin City University with **Kavya Kumar** and **Modi Eyobo**. *(Add a link to the report PDF here if you want to share it.)*

---

## What we set out to do

- Predict which customers will churn.
- Find the factors that matter most in that decision.
- Compare how four different models cope when churners are the minority class.
- Make the results explainable, not just accurate.

## The data

A telecom customer dataset with demographic, billing and subscription details. *(Add the source, the number of customers, and the share who churned, for example "7,043 customers, 26% churned". Readers need the churn rate to judge the scores below.)*

| Feature | Type |
|---|---|
| Age | Numerical |
| Gender | Categorical |
| Tenure | Numerical |
| MonthlyCharges | Numerical |
| TotalCharges | Numerical |
| Contract | Categorical |
| PaymentMethod | Categorical |
| Churn | Target |

## How we worked

**Preparing the data.** We handled missing values, label-encoded the categorical columns, scaled the numeric ones with `StandardScaler`, and split the data 80/20 with stratification so both sets kept the same churn rate.

**Models.** Logistic Regression, Decision Tree, Random Forest and XGBoost.

**Scoring.** We used F1 as the main measure, because with fewer churners than loyal customers, accuracy alone can look good while missing the people we care about. We also looked at precision, recall, confusion matrices and ROC curves.

**Going beyond the scores.**
- Statsmodels (p-values and AIC) to check the Logistic Regression features statistically
- Threshold tuning for Logistic Regression
- SHAP and feature importance to see what drives each prediction
- A train vs. test comparison to spot overfitting

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.677 | 0.509 | 0.716 | 0.595 |
| Decision Tree | 0.753 | 0.630 | 0.620 | **0.625** |
| Random Forest | 0.731 | 0.621 | 0.485 | 0.544 |
| XGBoost | 0.725 | 0.574 | 0.661 | 0.614 |

**What stood out**

- The **Decision Tree** had the best F1 (0.625), with XGBoost close behind at 0.614. The scores are modest, which is typical when the features are limited and the classes are imbalanced.
- **Logistic Regression** found the most churners (recall 0.716) at the cost of precision, and was the easiest model to interpret statistically.
- **Random Forest** overfitted noticeably and had the lowest recall. *(Add the train vs. test numbers here, for example "train F1 0.xx vs. test F1 0.xx".)*
- **XGBoost** predicted well and, with SHAP, gave explanations we could trust.

## What drives churn

Three factors came up at the top of the list in every model we checked:

1. **Contract type**
2. **Monthly charges**
3. **Tenure**

Customers on shorter contracts, paying more each month, and early in their time with the company, were the most likely to leave. *(Check the direction against your SHAP plots before keeping this sentence, then add one or two plots below.)*

### SHAP analysis

*(Add your SHAP summary plot here, plus two or three sentences on what it shows. The report has this analysis, but the README needs the picture.)*

---

## Limitations

- F1 around 0.6 means the models are useful for ranking risk, not for confident individual predictions.
- Results depend on one dataset with a small set of features.
- Logistic Regression's threshold was tuned on the same split we evaluated on. Cross-validated tuning would be more reliable.

## What we would do next

- Hyperparameter optimisation and cross-validated threshold tuning
- A deep learning approach for tabular data such as TabNet
- LIME and counterfactual explanations ("what would need to change for this customer to stay?")
- Survival analysis to predict *when* a customer will leave, not just whether
- A real-time scoring pipeline

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Statsmodels · SHAP · Matplotlib · Seaborn

## Run it yourself

```bash
git clone https://github.com/Sumu28/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook
```

*(Edit this to match what is actually in the repo. If there is no `churn_prediction.py`, remove it from the instructions.)*

## Repository structure

*(Check this against the real folders before publishing.)*

```
data/          # dataset (or instructions for downloading it)
notebooks/     # analysis and modelling
models/        # saved models
outputs/       # confusion matrices, SHAP plots, feature importance
Customer_Churn_Report.pdf
requirements.txt
README.md
```

## Authors

Sumukha Sagar, Modi Eyobo , Kavya Kumar
School of Computing, Dublin City University

## License

MIT
