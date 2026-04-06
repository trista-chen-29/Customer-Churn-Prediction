from fastapi import FastAPI
from app.schema import CustomerInput, PredictionResponse
from app.utils import get_risk_level, get_recommendation
import random
import joblib
import os

app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")

# Global model variable
# If the trained model file exists, it will be loaded at startup
# Otherwise, the API will use temporary dummy predictions
model = None


@app.on_event("startup")
def load_model():
    """Load trained model at startup if available"""
    global model
    model_path = "models/churn_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print("Model not found. Using dummy predictions for now.")


@app.get("/")
def root():
    """Health check endpoint to confirm the API is running."""
    return {"message": "Customer Churn API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):
    """
    Predict customer churn risk

    For now, if the real trained model is not available,
    the API returns a dummy prediction.
    """
    if model is None:
        prob = round(random.uniform(0, 1), 2)
    else:
        # Replace this later with real preprocessing + model inference
        prob = 0.5

    risk = get_risk_level(prob)
    recommendation = get_recommendation(prob)

    return {
        "churn_probability": prob,
        "risk_level": risk,
        "recommendation": recommendation
    }
