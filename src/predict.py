from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import clean_data, create_features


BASE_DIR = Path(__file__).resolve().parents[1]

CHURN_MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
SPEND_MODEL_PATH = BASE_DIR / "models" / "spend_model.joblib"

CHURN_THRESHOLD = 0.35


def _load_model(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def _prepare_customer(customer_data, task):
    customer_df = pd.DataFrame([customer_data])
    customer_df = clean_data(customer_df)

    # The API input intentionally does not contain either target.
    return create_features(customer_df, task=task)


def predict_churn(customer_data):
    model = _load_model(CHURN_MODEL_PATH)
    customer_df = _prepare_customer(customer_data, task="classification")

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(probability >= CHURN_THRESHOLD)

    return {
        "churn_probability": probability,
        "churn_prediction": prediction,
    }


def predict_spend(customer_data):
    model = _load_model(SPEND_MODEL_PATH)
    customer_df = _prepare_customer(customer_data, task="regression")

    prediction = max(0.0, float(model.predict(customer_df)[0]))

    return {
        "monthly_spend_prediction_pln": prediction,
    }


def predict_customer(customer_data):
    """Return both model outputs from the same customer payload."""
    return {
        **predict_churn(customer_data),
        **predict_spend(customer_data),
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
        "auto_renew": 1,
    }

    print(predict_customer(customer))
