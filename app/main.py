from fastapi import FastAPI
from app.schema import CustomerInput
from app.utils import get_risk_level, get_recommendation
import random
import joblib
import os

app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")

model = None


@app.on_event("startup")
def load_model():
    global model
    model_path = "models/churn_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print("Model not found. Using dummy predictions for now.")


@app.get("/")
def root():
    return {"message": "Customer Churn API is running"}


@app.post("/predict")
def predict(data: CustomerInput):
    # Temporary fallback until real model is ready
    if model is None:
        prob = round(random.uniform(0, 1), 2)
    else:
        # Replace this later with real preprocessing + prediction
        prob = 0.5

    risk = get_risk_level(prob)
    recommendation = get_recommendation(prob)

    return {
        "input": data.model_dump(),
        "churn_probability": prob,
        "risk_level": risk,
        "recommendation": recommendation
    }
