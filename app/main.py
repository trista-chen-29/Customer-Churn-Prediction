from fastapi import FastAPI
from app.schema import CustomerInput, PredictionResponse
from app.utils import get_risk_level, get_recommendation
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")

# Load the full trained pipeline.
# This pipeline already includes preprocessing, encoding, scaling, and model inference.
pipeline = joblib.load("models/churn_pipeline.pkl")


@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"message": "Customer Churn API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):
    """
    Predict churn probability for a customer using the trained pipeline.
    """
    input_df = pd.DataFrame([data.model_dump()])
    prob = float(pipeline.predict_proba(input_df)[0][1])

    risk = get_risk_level(prob)
    recommendation = get_recommendation(prob)

    return {
        "churn_probability": prob,
        "risk_level": risk,
        "recommendation": recommendation
    }
