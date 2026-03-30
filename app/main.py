from fastapi import FastAPI
from app.schema import CustomerInput
from app.utils import get_risk_level, get_recommendation
import random

app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")


@app.get("/")
def root():
    return {"message": "Customer Churn API is running"}


@app.post("/predict")
def predict(data: CustomerInput):
    prob = round(random.uniform(0, 1), 2)
    risk = get_risk_level(prob)
    recommendation = get_recommendation(prob)

    return {
        "input": data.model_dump(),
        "churn_probability": prob,
        "risk_level": risk,
        "recommendation": recommendation
    }