from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_customer


app = FastAPI(
    title="Customer Analytics Prediction API",
    description="API for predicting customer churn and monthly spend",
    version="2.0.0",
)


class Customer(BaseModel):
    age: int
    city: str
    plan: str
    acquisition_channel: str
    tenure_months: int = Field(ge=0)
    satisfaction_score: float
    monthly_income_pln: float
    support_tickets_last_30d: int = Field(ge=0)
    avg_logins_last_30d: float = Field(ge=0)
    discount_pct: float = Field(ge=0)
    auto_renew: int


@app.get("/")
def root():
    return {
        "message": "Customer Analytics Prediction API is running",
        "models": ["churn", "monthly_spend"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    return predict_customer(customer.model_dump())
