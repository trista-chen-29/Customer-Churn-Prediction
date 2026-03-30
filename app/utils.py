def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"


def get_recommendation(prob: float) -> str:
    if prob < 0.3:
        return "Customer is likely to stay. Maintain engagement."
    elif prob < 0.7:
        return "Offer small loyalty incentives."
    return "Offer discount or contract upgrade."