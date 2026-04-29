# Customer Churn Prediction (In Progress)

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


### Features
- Demographics (gender, senior citizen, partner, dependents)  
- Account info (tenure, contract, billing method)  
- Services (internet, streaming, tech support, etc.)  
- Financial data (monthly & total charges)  

### Target Variable:
- **Churn** (Yes/No)

### Additional Dataset (Extension)
- **250K Customer Churn Dataset**  
- https://www.kaggle.com/datasets/rhythmghai/250k-customer-churn-prediction-dataset  

Used to evaluate model performance on a larger dataset.

---

## Approach

### 1. Data Preprocessing
- Handle missing values  
- Convert `TotalCharges` to numeric  
- Remove invalid rows  
- Encode categorical variables  
- Scale features for logistic regression

### 2. Feature Engineering
- `AvgCharges = TotalCharges / tenure`  
- `NumServices = number of services used`  
- `ProtectionCount = number of protection-related services`  

### 3. Model Training
- Logistic Regression  
- Random Forest  

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-score  
- Confusion Matrix  
- ROC-AUC  

### 5. Deployment
A FastAPI backend is used to:
- Accept customer input  
- Run predictions using a trained pipeline  
- Return churn probability and recommendations  

---

## Exploratory Data Analysis (EDA)

### Key Observations:
- Customers with **month-to-month** contracts have the highest churn
- Customers with **low tenure** customers are more likely to churn
- Customers with **higher monthly charges** show higher churn tendency
- Customers without **tech support**, **online security**, and **device protection** are more likely to churn
- **Fiber optic** users show higher churn behavior

See: `notebooks/eda.ipynb`

---

## Model Development

### Models Trained

#### Logistic Regression
- Uses scaled features
- Achieved slightly better recall
- Better suited for identifying customers at risk of churn

#### Random Forest
- Does not require scaling
- Captures non-linear relationships
- Achieved slightly higher precision

### Model Results

| Model | Precision | Recall | F1-score | ROC-AUC |
|------|-----------|--------|----------|---------|
| Logistic Regression | 0.62 | 0.51 | 0.56 | 0.83 |
| Random Forest | 0.66 | 0.48 | 0.56 | 0.83 |

### Key Insights:
- Both models achieved similar overall performance (F1 ≈ 0.56, ROC-AUC ≈ 0.83)
- Logistic Regression has higher recall, meaning it detects more churn customers
- Random Forest has higher precision, meaning fewer false positives
- For churn prediction, recall is more important, making Logistic Regression the preferred model

### Final Model Choice
**Logistic Regression** was selected for API deployment because recall is especially important in churn prediction, where identifying at-risk customers matters more than minimizing false positives.

---

## Technologies Used

- Python
- Scikit-learn
- FastAPI
- Pandas
- NumPy
- Matplotlib

---

## Deployment (FastAPI)

We saved a **complete scikit-learn pipeline** (`churn_pipeline.pkl`) that includes:
- preprocessing  
- encoding  
- scaling  
- trained model  

The backend loads this pipeline and performs real-time predictions.

---

## API 

### Endpoint

`POST /predict`

### Example input

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

## Project Structure

```text
Customer-Churn-Prediction/
├── app/                    # FastAPI backend
├── data/                   # Dataset files
├── models/                 # Saved model (pipeline)
├── notebooks/              # EDA
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
- Tested API with real predictions

---

## Extended Experiment

We will train a model on a larger dataset (250K records) to:
- compare performance
- evaluate scalability

### Result



---

## Future Work

- Hyperparameter tuning
- Improve validation and UX
- Add frontend interface
- Deploy application
