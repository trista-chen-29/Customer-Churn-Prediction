from pydantic import BaseModel

class CustomerInput(BaseModel):
    tenure: int
    monthly_charges: float
    contract: str
    internet_service: str