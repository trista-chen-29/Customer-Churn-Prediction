FINAL_COLUMN_ORDER = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'AvgCharges',
    'NumServices',
    'ProtectionCount',
    'gender_Male',
    'SeniorCitizen_Yes',
    'Partner_Yes',
    'Dependents_Yes',
    'PhoneService_Yes',
    'MultipleLines_No phone service',
    'MultipleLines_Yes',
    'InternetService_Fiber optic',
    'InternetService_No',
    'OnlineSecurity_No internet service',
    'OnlineSecurity_Yes',
    'OnlineBackup_No internet service',
    'OnlineBackup_Yes',
    'DeviceProtection_No internet service',
    'DeviceProtection_Yes',
    'TechSupport_No internet service',
    'TechSupport_Yes',
    'StreamingTV_No internet service',
    'StreamingTV_Yes',
    'StreamingMovies_No internet service',
    'StreamingMovies_Yes',
    'Contract_One year',
    'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]


def get_risk_level(prob: float) -> str:
    """
    Convert churn probability into a risk category.
    """
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"


def get_recommendation(prob: float) -> str:
    """
    Return a simple retention recommendation based on churn probability.
    """
    if prob < 0.3:
        return "Customer is likely to stay. Maintain engagement."
    elif prob < 0.7:
        return "Offer small loyalty incentives."
    return "Offer discount or contract upgrade."


def preprocess_input(data):
    """
    Transform API input into the exact feature order expected by the model.
    """
    features = {col: 0 for col in FINAL_COLUMN_ORDER}

    # Numeric features
    features["tenure"] = data.tenure
    features["MonthlyCharges"] = data.MonthlyCharges
    features["TotalCharges"] = data.TotalCharges
    features["AvgCharges"] = data.TotalCharges / data.tenure if data.tenure > 0 else 0

    # Service counts
    yes_services = [
        data.PhoneService,
        data.MultipleLines,
        data.OnlineSecurity,
        data.OnlineBackup,
        data.DeviceProtection,
        data.TechSupport,
        data.StreamingTV,
        data.StreamingMovies,
    ]
    features["NumServices"] = sum(value == "Yes" for value in yes_services)

    protection_services = [
        data.OnlineSecurity,
        data.OnlineBackup,
        data.DeviceProtection,
        data.TechSupport,
    ]
    features["ProtectionCount"] = sum(value == "Yes" for value in protection_services)

    # Dummy variables
    if data.gender == "Male":
        features["gender_Male"] = 1

    if data.SeniorCitizen == "Yes":
        features["SeniorCitizen_Yes"] = 1

    if data.Partner == "Yes":
        features["Partner_Yes"] = 1

    if data.Dependents == "Yes":
        features["Dependents_Yes"] = 1

    if data.PhoneService == "Yes":
        features["PhoneService_Yes"] = 1

    if data.MultipleLines == "No phone service":
        features["MultipleLines_No phone service"] = 1
    elif data.MultipleLines == "Yes":
        features["MultipleLines_Yes"] = 1

    if data.InternetService == "Fiber optic":
        features["InternetService_Fiber optic"] = 1
    elif data.InternetService == "No":
        features["InternetService_No"] = 1

    if data.OnlineSecurity == "No internet service":
        features["OnlineSecurity_No internet service"] = 1
    elif data.OnlineSecurity == "Yes":
        features["OnlineSecurity_Yes"] = 1

    if data.OnlineBackup == "No internet service":
        features["OnlineBackup_No internet service"] = 1
    elif data.OnlineBackup == "Yes":
        features["OnlineBackup_Yes"] = 1

    if data.DeviceProtection == "No internet service":
        features["DeviceProtection_No internet service"] = 1
    elif data.DeviceProtection == "Yes":
        features["DeviceProtection_Yes"] = 1

    if data.TechSupport == "No internet service":
        features["TechSupport_No internet service"] = 1
    elif data.TechSupport == "Yes":
        features["TechSupport_Yes"] = 1

    if data.StreamingTV == "No internet service":
        features["StreamingTV_No internet service"] = 1
    elif data.StreamingTV == "Yes":
        features["StreamingTV_Yes"] = 1

    if data.StreamingMovies == "No internet service":
        features["StreamingMovies_No internet service"] = 1
    elif data.StreamingMovies == "Yes":
        features["StreamingMovies_Yes"] = 1

    if data.Contract == "One year":
        features["Contract_One year"] = 1
    elif data.Contract == "Two year":
        features["Contract_Two year"] = 1

    if data.PaperlessBilling == "Yes":
        features["PaperlessBilling_Yes"] = 1

    if data.PaymentMethod == "Credit card (automatic)":
        features["PaymentMethod_Credit card (automatic)"] = 1
    elif data.PaymentMethod == "Electronic check":
        features["PaymentMethod_Electronic check"] = 1
    elif data.PaymentMethod == "Mailed check":
        features["PaymentMethod_Mailed check"] = 1

    return [features[col] for col in FINAL_COLUMN_ORDER]
