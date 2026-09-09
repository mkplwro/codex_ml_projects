from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import (
    clean_data,
    create_features,
    create_preprocessor
)


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "subscription_customers_dirty.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"


def train_model():
    # Load data
    data = pd.read_csv(DATA_PATH)

    # Clean data
    data = clean_data(data)

    # Remove rows without target values
    data = data.dropna(subset=["churned"])

    # Create features
    X = create_features(data)
    y = data["churned"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Create model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
