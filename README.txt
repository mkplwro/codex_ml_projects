# Customer Monthly Spend Prediction

## Project Overview
What is being predicted and why?

## Dataset
Number of observations, features and data quality issues.

## Methodology
Data cleaning → preprocessing → CV → model selection → evaluation.

## Models Compared
Table with CV MAE / RMSE / R².

## Key Findings
- Lasso performed best in cross-validation.
- Complex models did not outperform regularized linear models.
- Extreme target values substantially affected evaluation.
- The holdout test set was easier because it did not contain extreme observations.

## Final Results
Emphasize CV performance, not only test performance.

## Limitations
- only five extreme observations
- uncertain data quality
- representativeness of holdout split
- possible temporal issues

## Tech Stack
Python, pandas, scikit-learn, XGBoost, Optuna.