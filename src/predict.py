from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import clean_data, create_features


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    model = load_model()

    customer_df = pd.DataFrame([customer_data])

    customer_df = clean_data(customer_df)
    customer_df = create_features(
        customer_df.assign(churned=0, monthly_spend_pln=0)
    )

    probability = model.predict_proba(customer_df)[0, 1]

    prediction = int(probability >= 0.35)

    return {
        "churn_probability": float(probability),
        "churn_prediction": prediction
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

    result = predict_churn(customer)

    print(result)