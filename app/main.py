from fastapi import FastAPI
from app.schema import CustomerInput, PredictionResponse
from app.utils import get_risk_level, get_recommendation, preprocess_input, FINAL_COLUMN_ORDER
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")

# Load trained model and scaler
model = joblib.load("models/logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.get("/")
def root():
    """
    Health check endpoint to confirm the API is running.
    """
    return {"message": "Customer Churn API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):
    """
    Predict customer churn risk using the trained logistic regression model.
    """
    processed = preprocess_input(data)
    input_df = pd.DataFrame([processed], columns=FINAL_COLUMN_ORDER)
    scaled = scaler.transform(input_df)
    prob = float(model.predict_proba(scaled)[0][1])

    risk = get_risk_level(prob)
    recommendation = get_recommendation(prob)

    return {
        "churn_probability": prob,
        "risk_level": risk,
        "recommendation": recommendation
    }
