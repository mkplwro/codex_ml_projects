from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn Prediction API is running"


def test_predict():
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

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "churn_prediction" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]