from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability",
    version="1.0.0"
)


class Customer(BaseModel):
    age: int
    city: str
    plan: str
    acquisition_channel: str
    tenure_months: int
    satisfaction_score: float
    monthly_income_pln: float
    support_tickets_last_30d: int
    avg_logins_last_30d: float
    discount_pct: float
    auto_renew: int


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: Customer):
    result = predict_churn(customer.model_dump())
    return result