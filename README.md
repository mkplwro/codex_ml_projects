# Customer Monthly Spend Prediction

## Project Overview

Caution! This is my first project and dirty data was made by LLM to practice and train my skills.

This project is an end-to-end machine learning regression project focused on predicting customer monthly spending in PLN.

The main goal was to build, compare, tune, and evaluate regression models using customer-related data. The project also focuses on an important part of machine learning: understanding whether model evaluation results are reliable.

The workflow includes data cleaning, exploratory data analysis, feature engineering, preprocessing, model comparison, hyperparameter tuning, cross-validation, repeated cross-validation, model diagnostics, and error analysis.

The final analysis showed that model performance strongly depends on how extreme target values are distributed across validation folds. This became one of the most important findings of the project.

---

## Business Problem

Customer spending is an important metric for many subscription-based businesses. Predicting how much a customer is likely to spend can help identify high-value customers and better understand which characteristics are related to customer revenue.

The objective of this project is to predict:

`monthly_spend_pln`

using available customer information.



The dataset contains customer-related variables such as age, country, subscription information, monthly income, customer tenure, payment information, and auto-renewal status.

---

## Data Preparation

The dataset was checked for common data quality issues, including:

* duplicate records
* missing values
* incorrect data types
* categorical variables
* unusual observations
* extreme target values

Identifier columns were removed because they do not provide useful information for general predictions.

The `churned` feature was excluded from the final model because it may represent information that is not available at the time of prediction. Including such information could potentially introduce data leakage.

The preprocessing workflow was built using a scikit-learn Pipeline. Numerical features were processed using imputation and scaling, while categorical features were processed using imputation and one-hot encoding.

Using a Pipeline helps prevent data leakage because preprocessing is performed separately inside each cross-validation fold.

---

## Feature Engineering

A new feature called `income_per_tenure` was created by combining monthly income and customer tenure:

```text
monthly_income_pln / (tenure_months + 1)
```

The purpose of this feature was to test whether the relationship between customer income and the length of the customer relationship could provide additional predictive information.

Feature engineering was evaluated as part of the machine learning workflow rather than being assumed to improve the model automatically.

---

## Models Compared

Several regression models were compared using cross-validation:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Elastic Net
* Decision Tree Regressor
* XGBoost Regressor

Model performance was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² score

Lasso Regression was selected as the final model because regularized linear models performed competitively compared to more complex models.

Hyperparameter tuning was also performed using GridSearchCV and Optuna.

One important result was that more complex models did not automatically provide better performance. This shows that a simpler regularized model can sometimes generalize better than more complex algorithms.

---

## Model Evaluation

The initial evaluation produced very different results between standard cross-validation and the holdout test set.

| Metric | Cross-Validation | Test Set |
| ------ | ---------------: | -------: |
| MAE    |            13.24 |    11.45 |
| RMSE   |            56.81 |    14.57 |
| R²     |             0.36 |     0.87 |

The test results were significantly better than the cross-validation results. Further analysis showed that the most extreme target observations were not present in the holdout test set.

This means that the test set was easier to predict and the test score is likely optimistic. For this reason, the test R² score should not be interpreted as the main estimate of the model's expected performance.

---

## Repeated Cross-Validation

To obtain a more stable estimate of model performance, Repeated K-Fold Cross-Validation was performed using 10 folds repeated 10 times.

The results were:

| Metric        | Result |
| ------------- | -----: |
| MAE           |  13.22 |
| RMSE (mean)   |  46.51 |
| RMSE (median) |  15.36 |
| RMSE (std)    |  40.01 |
| RMSE (min)    |  13.78 |
| RMSE (max)    | 145.43 |
| R²            |   0.57 |

The large difference between the mean and median RMSE suggests that most validation folds had relatively low prediction errors, while a small number of folds produced very large errors.

The high RMSE standard deviation also shows that model performance depends strongly on the data split. This is likely related to extreme target values appearing in some validation folds and strongly increasing the prediction error.

Repeated cross-validation does not improve the model itself. Instead, it provides a more robust estimate of how the model behaves across different data splits.

---

## Extreme Target Values

The dataset contains a small number of observations with monthly spending equal to 2500 PLN.

These observations were not removed automatically because extreme values are not necessarily incorrect. Removing observations only because they worsen model performance could create misleading evaluation results.

Instead, the extreme observations were treated as an important part of the analysis.

The large difference between MAE and RMSE suggests that the model performs reasonably well for most customers but makes much larger errors for a small number of observations.

RMSE is particularly sensitive to large errors, which explains why a small number of extreme cases can strongly affect the average RMSE.

---

## Key Findings

The main findings from this project are:

1. Lasso Regression performed competitively against more complex models.
2. A high score from a single holdout test set can be misleading when the target distribution is not representative.
3. Repeated cross-validation provided a more reliable estimate of model performance.
4. The model performs relatively well for most observations but struggles with a small number of extreme cases.
5. Extreme target values have a strong influence on RMSE and model stability.
6. Model evaluation should include multiple metrics and should not rely on a single train/test split.

---

## Technologies Used

* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* XGBoost
* Optuna

---

## Future Improvements

Possible improvements include:

* further investigation of extreme customer spending values
* testing target transformations such as `log1p`
* comparing performance with a DummyRegressor baseline
* testing additional feature engineering approaches
* improving the representation of extreme observations during validation
* adding model interpretation methods
* collecting more data for high-value customers

---

## Project Structure

```text
project/
│
├── data/
│   └── subscription_customers_dirty.csv
│
├── notebooks/
│   └── regression_model.ipynb
│   └── classification_model.ipyng
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Conclusion

This project demonstrates a complete machine learning regression workflow, from data preparation to model evaluation and error analysis.

The most important lesson from this project was that machine learning metrics should not be interpreted without understanding the data behind them. Although the holdout test set produced a high R² score, repeated cross-validation showed that model performance was less stable across different data splits.

This project highlights the importance of robust validation, multiple evaluation metrics, and investigating extreme observations instead of focusing only on achieving the highest possible score.
