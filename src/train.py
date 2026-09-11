from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    clean_data,
    create_features,
    create_preprocessor,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "subscription_customers_dirty.csv"
MODELS_DIR = BASE_DIR / "models"

CHURN_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
SPEND_MODEL_PATH = MODELS_DIR / "spend_model.joblib"

# Selected in regression_model.ipynb via GridSearchCV.
SPEND_LASSO_ALPHA = 0.2
CHURN_THRESHOLD = 0.35


def load_training_data():
    data = pd.read_csv(DATA_PATH)
    return clean_data(data)


def train_churn(data):
    data = data.dropna(subset=["churned"])

    X = create_features(data, task="classification")
    y = data["churned"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    joblib.dump(model, CHURN_MODEL_PATH)

    return model


def train_spend(data):
    # A negative monthly spend is not a valid production target.
    data = data.dropna(subset=["monthly_spend_pln"])
    data = data[data["monthly_spend_pln"] >= 0]

    X = create_features(data, task="regression")
    y = data["monthly_spend_pln"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Lasso(alpha=0.2) is the final model selected in regression_model.ipynb.
    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "regressor",
                Lasso(
                    alpha=SPEND_LASSO_ALPHA,
                    max_iter=10000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    joblib.dump(model, SPEND_MODEL_PATH)

    return model


def train_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_training_data()
    train_churn(data)
    train_spend(data)

    print("Models trained successfully.")
    print(f"Churn model: {CHURN_MODEL_PATH}")
    print(f"Spend model: {SPEND_MODEL_PATH}")


if __name__ == "__main__":
    train_models()
