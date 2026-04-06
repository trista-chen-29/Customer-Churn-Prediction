from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Input schema for customer churn prediction"""
    tenure: int = Field(..., ge=0, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount")
    contract: str = Field(..., description="Contract type")
    internet_service: str = Field(..., description="Type of internet service")


class PredictionResponse(BaseModel):
    """Response schema returned by the prediction API"""
    churn_probability: float
    risk_level: str
    recommendation: str
