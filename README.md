# Customer Analytics & Behavior Prediction Suite


---

##This project presents an end-to-end Machine Learning ecosystem focused on subscription-based customer analytics using a synthetic dataset containing realistic data quality challenges. The portfolio combines two complementary predictive modeling tasks designed to solve critical business problems:

- Customer Churn Prediction (Classification): Identifying customers at risk of leaving the service to enable proactive retention interventions.

- Customer Monthly Spend Prediction (Regression): Estimating monthly revenue per customer (monthly_spend_pln) to pinpoint high-value accounts and analyze financial drivers.

Both sub-projects focus on establishing clean data pipelines, robust cross-validation schemes, strict data leakage prevention, and critical evaluation of model behavior under real-world data noise.

---

##Business Problem & Objectives

In subscription business models, optimizing Customer Lifetime Value requires addressing both customer retention and spend maximization:

Churn Classification: Failing to detect a churning customer (False Negative) is significantly more costly than a false alarm (False Positive). Therefore, the modeling strategy prioritizes Recall and the F2 score over raw Accuracy.

Spend Regression: Understanding expected monthly spending helps prioritize key accounts and optimize marketing channels. The goal is to build reliable regression baselines and evaluate their performance consistency.

---

## Dataset & Data Quality Management

The dataset consists of approximately 10,000 customer records featuring demographic details, subscription tiers, support interaction counts, satisfaction scores, auto-renewal flags, and financial statistics.

To simulate real-world conditions, the raw data contains intentional anomalies:

- Dirty Numerical Values: Negative incomes, invalid ages, and sentinel numbers (e.g., 999999) were replaced with NaN and processed via imputation.

- Categorical Inconsistencies: Standardized via whitespace stripping and lowercase conversion.

- Duplicates & Leakage Risk: Customer ID duplicates were evaluated to distinguish lifecycle updates from redundant rows. Standard identifier columns were dropped to prevent overfitting.

- Target Anomalies: Extreme monthly spend outliers (e.g., 2500 PLN) were kept deliberately to analyze model robustness rather than masking real edge cases.
---

## Machine Learning Workflow

Both pipelines follow a strict, leakage-free structure utilizing scikit-learn Pipelines:

- Data Cleaning & Preprocessing: Numerical features use median imputation and scaling; categorical features undergo imputation and one-hot encoding.

- Feature Engineering:
Classification: satisfaction_per_tenure = satisfaction_score / (tenure_months + 1)

Regression: income_per_tenure = monthly_income_pln / (tenure_months + 1)

- Model Selection & Tuning:
Classification: Evaluation of tree-based models and classifiers optimized via stratified cross-validation and custom decision thresholds.

Regression: Comparison of Linear, Ridge, Lasso, ElasticNet, Decision Tree, and XGBoost models using GridSearchCV and Optuna. Regularized models (Lasso) demonstrated superior generalization compared to overly complex architectures.

---

## Key Insights & Validation Findings

Metric Realism over High Scores: In the regression task, a single holdout test set yielded an optimistic $R^2$ of 0.87. However, 10-fold Repeated Cross-Validation revealed a mean $R^2$ of 0.57 and a wide RMSE spread (mean: 46.51, median: 15.36). This confirmed that extreme spend values in specific validation folds heavily skew evaluation—highlighting why single train-test splits can be misleading.

Class Imbalance & Threshold Tuning: Strategic threshold adjustment in churn classification significantly improved retention detection (F2/Recall) over standard default cutoff boundaries.
---


## Production MLOps layer

The production API now serves **two independent models** from the same customer payload:

- `churn_model.joblib` — binary classification of churn risk.
- `spend_model.joblib` — regression of `monthly_spend_pln`.

The regression model is the **Lasso model with `alpha=0.2` selected by GridSearchCV in `notebooks/regression_model.ipynb`**. The production training code keeps the same preprocessing logic and task-specific feature engineering as the notebooks.

### Train both models

```bash
python -m src.train
```

This creates:

```text
models/
├── churn_model.joblib
└── spend_model.joblib
```

### Run the API

```bash
uvicorn src.api:app --reload
```

`POST /predict` returns:

```json
{
  "churn_probability": 0.72,
  "churn_prediction": 1,
  "monthly_spend_prediction_pln": 184.37
}
```

The API never accepts either target as an input, so `churned` and `monthly_spend_pln` cannot leak into inference.

### Regression target handling

The source data contains three negative `monthly_spend_pln` values. These are excluded from regression training because negative customer spend is not a valid production target. The five extreme `2500 PLN` observations are retained so that the model is not artificially trained only on the easy part of the target distribution.

The notebook's repeated cross-validation should remain the primary performance reference: the single holdout test split is optimistic because all extreme spend observations fall into the training set.

## Technologies Used

Languages & Core: Python, NumPy, pandas

Modeling & Optimization: scikit-learn, XGBoost, Optuna

Visualization: Matplotlib
---

## Project Structure

project/
├── data/
│   └── subscription_customers_dirty.csv
├── models/
│   ├── churn_model.joblib
│   └── spend_model.joblib
├── notebooks/
│   ├── classification_model.ipynb
│   └── regression_model.ipynb
├── src/
│   ├── api.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   └── test_api.py
├── README.md
└── requirements.txt