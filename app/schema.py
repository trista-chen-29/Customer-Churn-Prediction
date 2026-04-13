from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


class PredictionResponse(BaseModel):
    """
    Response schema returned by the prediction API.
    """
    churn_probability: float
    risk_level: str
    recommendation: str
