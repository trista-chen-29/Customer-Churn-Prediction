def get_risk_level(prob: float) -> str:
    """Convert churn probability into a risk category"""
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"


def get_recommendation(prob: float) -> str:
    """Return a simple retention recommendation based on churn probability"""
    if prob < 0.3:
        return "Customer is likely to stay. Maintain engagement."
    elif prob < 0.7:
        return "Offer small loyalty incentives."
    return "Offer discount or contract upgrade."


def preprocess_input(data):
    """
    Placeholder for future preprocessing

    This function will later transform API input into the
    format expected by the trained machine learning model
    """
    return [
        data.tenure,
        data.monthly_charges
    ]
