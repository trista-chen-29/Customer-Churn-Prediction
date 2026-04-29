from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """
    Input schema for customer churn prediction.
    """
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: str = Field(..., description="Whether the customer is a senior citizen: Yes or No")
    Partner: str = Field(..., description="Whether the customer has a partner: Yes or No")
    Dependents: str = Field(..., description="Whether the customer has dependents: Yes or No")
    tenure: int = Field(..., ge=0, description="Customer tenure in months")
    PhoneService: str = Field(..., description="Whether the customer has phone service: Yes or No")
    MultipleLines: str = Field(..., description="Multiple lines status")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Online security service status")
    OnlineBackup: str = Field(..., description="Online backup service status")
    DeviceProtection: str = Field(..., description="Device protection status")
    TechSupport: str = Field(..., description="Tech support status")
    StreamingTV: str = Field(..., description="Streaming TV subscription status")
    StreamingMovies: str = Field(..., description="Streaming movies subscription status")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Paperless billing: Yes or No")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")


class PredictionResponse(BaseModel):
    """
    Response schema returned by the prediction API.
    """
    churn_probability: float
    risk_level: str
    recommendation: str
