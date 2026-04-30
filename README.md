# Customer Churn Prediction

## Team Members
- Trista Chen — trista.chen@sjsu.edu  
- Sonal Rana — sonal.rana@sjsu.edu  

---

## Problem Statement
Customer churn is a major challenge for telecom companies, as losing existing customers leads to significant revenue loss and higher acquisition costs.

This project builds a machine learning system that predicts whether a telecom customer is likely to churn based on demographic information, service usage, and account details. The goal is to support data-driven retention strategies.

---

## Dataset

### Primary Dataset
- **Telco Customer Churn Dataset (Kaggle)**  
- https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction  

### Target Variable
- **Churn (Yes/No)**

### Additional Dataset (Extension)
- **250K Customer Churn Dataset**  
- https://www.kaggle.com/datasets/rhythmghai/250k-customer-churn-prediction-dataset  

Used to evaluate model performance on a larger dataset.

---

## Approach

### Data Preprocessing
- Handle missing values  
- Convert `TotalCharges` to numeric  
- Remove invalid rows  
- Encode categorical variables  
- Scale features for logistic regression  

### Feature Engineering
- `AvgCharges = TotalCharges / tenure`  
- `NumServices = number of services used`  
- `ProtectionCount = number of protection-related services`  

### Models Trained
- Logistic Regression  
- Random Forest Classifier
- Gradient Boosting Classifier
---

## Exploratory Data Analysis (EDA)

### Key Observations
- Month-to-month contracts have the highest churn  
- Low tenure customers churn more  
- Higher monthly charges increase churn likelihood  
- Lack of tech support/security correlates with churn  
- Fiber optic users show higher churn behavior  

See: `notebooks/eda.ipynb`

---

### Model Results (Baseline Dataset)

| Model | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Logistic Regression | 0.50 | 0.79 | 0.52 |
| Random Forest | 0.63 | 0.44 | 0.52 |
| Gradient Boost | .64 | .48 | 0.55 | 

### Key Insights:
- Logistic Regression performed best in terms of recall (79%), it correctly identified most customers who actually churned. 
- This is important because minimizing missed churn cases (false negatives) is critical in customer retention scenarios.
- Random Forest and Gradient Boosting achieved higher precision (63% and 64%), meaning their churn predictions were more       accurate when they did predict churn, but they missed a larger portion of actual churners (higher false negatives).

### Final Model Choice
**Logistic Regression** was selected for API deployment because recall is especially important in churn prediction, where identifying at-risk customers matters more than minimizing false positives.

---

## Deployment (FastAPI)

A FastAPI backend is used to serve predictions.

We saved a **complete scikit-learn pipeline (`churn_pipeline.pkl`)** that includes:
- preprocessing  
- encoding  
- scaling  
- trained model  

The API uses this pipeline for real-time inference.

---

## API 

### Endpoint
`POST /predict`

### Example Input
```json
{
  "gender": "Male",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.99,
  "TotalCharges": 959.88
}
```

### Example output
```json
{
  "churn_probability": 0.60,
  "risk_level": "Medium",
  "recommendation": "Offer small loyalty incentives."
}
```

---

## Extended Experiment (250K Dataset)

To evaluate scalability, we trained the same model on a larger dataset with 250,000 records.

### Result
- Accuracy: 0.83
- Precision (churn): 0.74
- Recall (churn): 0.70
- F1-score: 0.72
- ROC-AUC: 0.90

### Comparison
| Metric   | Baseline | 250K Dataset |
| -------- | -------- | ------------ |
| F1-score | 0.52     | **0.72**     |
| Recall   | 0.79     | **0.70**     |
| ROC-AUC  | --     | **0.90**     |

### Key Insight
Performance improved significantly with more data, indicating that the model generalizes better and benefits from increased dataset size and richer features.

---

## Project Structure

```text
Customer-Churn-Prediction/
├── app/                    # FastAPI backend
├── data/                   # Dataset files
├── models/                 # Saved pipeline model
├── notebooks/              # EDA and experiments
├── src/                    # Training scripts
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

---

## Current Implementation Progress

- Completed EDA and feature engineering
- Trained and evaluated models
- Selected Logistic Regression
- Built and saved full pipeline
- Integrated model into FastAPI backend
- Tested API successfully
- Evaluated model on large-scale dataset

---

## Future Work

- Hyperparameter tuning
- Improve validation and UX
- Add frontend interface
- Deploy application
