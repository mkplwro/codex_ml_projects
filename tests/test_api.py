from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


CUSTOMER = {
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


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["models"] == ["churn", "monthly_spend"]


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_both_models():
    response = client.post("/predict", json=CUSTOMER)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "churn_prediction" in result
    assert "monthly_spend_prediction_pln" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]
    assert result["monthly_spend_prediction_pln"] >= 0
