from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_data, create_preprocessor


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "subscription_customers_dirty.csv"
MODEL_PATH = BASE_DIR / "models" / "spend_model.joblib"


def create_regression_features(data):
    data = data.copy()

    X = data.drop(
        columns=["monthly_spend_pln", "churned"],
        errors="ignore"
    )

    X["income_per_tenure"] = (
        X["monthly_income_pln"] / (X["tenure_months"] + 1)
    )

    return X


def train_regression_model():
    # Load data
    data = pd.read_csv(DATA_PATH)

    # Clean data
    data = clean_data(data)

    # Remove rows without target
    data = data.dropna(subset=["monthly_spend_pln"])

    # Create features
    X = create_regression_features(data)
    y = data["monthly_spend_pln"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Create regression pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                Lasso(
                    alpha=0.2,
                    max_iter=10000
                )
            )
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("Regression model trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_regression_model()