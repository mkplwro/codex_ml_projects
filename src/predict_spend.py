import joblib
import pandas as pd
from pathlib import Path

from src.preprocessing import clean_data


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "spend_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


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


def predict_spend(customer_data):
    model = load_model()

    customer_df = pd.DataFrame([customer_data])

    customer_df = clean_data(customer_df)
    customer_df = create_regression_features(customer_df)

    prediction = model.predict(customer_df)[0]

    return {
        "predicted_monthly_spend_pln": float(prediction)
    }


if __name__ == "__main__":
    customer = {
        "age": 35,
        "city": "wroclaw",
        "plan": "premium",
        "acquisition_channel": "online",
        "tenure_months": 12,
        "satisfaction_score": 7,
        "monthly_income_pln": 7000,
        "support_tickets_last_30d": 1,
        "avg_logins_last_30d": 15,
        "discount_pct": 10,
        "auto_renew": 1
    }

    result = predict_spend(customer)

    print(result)