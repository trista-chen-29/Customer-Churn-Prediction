# 📊 Customer Churn Prediction (In Progress)

## 👥 Team Members
- Trista Chen — trista.chen@sjsu.edu  
- Sonal Rana — sonal.rana@sjsu.edu  

---

## 🧩 Problem Statement
Customer churn is a major challenge for telecom companies, as losing existing customers leads to significant revenue loss and higher costs to acquire new customers.

This project builds a machine learning system that predicts whether a telecom customer is likely to churn based on demographic information, service usage, and account details. The goal is to support data-driven retention strategies and reduce customer loss.

---

## 📂 Dataset

### Primary Dataset
We used the **Telco Customer Churn** dataset from Kaggle:

- **Source:** Customer Churn Prediction Dataset  
- **Link:** https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction  

### Features include:
- Customer demographics (gender, senior citizen, partner, dependents)
- Account information (tenure, contract type, billing method)
- Services used (internet service, streaming, tech support, etc.)
- Financial data (monthly charges, total charges)

### Target Variable:
- **Churn** (Yes/No)

### Additional Dataset (Planned)
To extend this project beyond a basic baseline, we plan to experiment with a larger and more complex dataset:

- **Source:** 250K Customer Churn Prediction Dataset
- **Link:** https://www.kaggle.com/datasets/rhythmghai/250k-customer-churn-prediction-dataset  

This dataset will allow us to:
- Evaluate model performance on a larger, more realistic dataset  
- Compare results with the baseline dataset  
- Improve generalization and robustness of the model  

---

## ⚙️ Planned Approach

### 1. Data Preprocessing
- Handle missing values
- Convert `TotalCharges` to numeric
- Remove invalid rows
- Encode categorical features using one-hot encoding
- Scale features for logistic regression

### 2. Exploratory Data Analysis (EDA)
- Analyze patterns related to churn
- Visualize relationships between key features and churn behavior

### 3. Feature Engineering
The following new features were created:
- `AvgCharges` = `TotalCharges / tenure`
- `NumServices` = total number of services used
- `ProtectionCount` = number of protection-related services

### 4. Model Development
Two classification models were trained:
- **Logistic Regression**
- **Random Forest**

### 5. Deployment
A **FastAPI** backend was built to:
- Accept customer information as input
- Preprocess input data to match training features
- Return churn probability
- Classify customer risk level
- Provide a retention recommendation  

---

## 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns influencing churn.

### Key Observations:
- Customers with **month-to-month** contracts have the highest churn
- Customers with **low tenure** customers are more likely to churn
- Customers with **higher monthly charges** show higher churn tendency
- Customers without **tech support**, **online security**, and **device protection** are more likely to churn
- **Fiber optic** users show higher churn behavior

See: `notebooks/eda.ipynb`

---

## 🤖 Model Development

### Models Trained

#### Logistic Regression
- Uses scaled features
- Achieved slightly better recall
- Better suited for identifying customers at risk of churn

#### Random Forest
- Does not require scaling
- Captures non-linear relationships
- Achieved slightly higher precision

### Model Performance

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

## 🗂️ Project Structure

```text
Customer-Churn-Prediction/
│
├── app/                    # FastAPI backend
│   ├── main.py
│   ├── schema.py
│   ├── utils.py
│   └── tests/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/                 # Saved trained models
├── notebooks/              # EDA notebook
├── src/                    # Training notebook / scripts
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- FastAPI
- Pandas
- NumPy
- Matplotlib

---

## ▶️ How to Run

### 1. Create and activate the environment
Use the Python 3.11 virtual environment:

```bash
python3.11 -m venv venv311
source venv311/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Then open:
```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### **POST** `/predict`

This endpoint accepts customer information and returns:
- churn probability
- risk level
- retention recommendation

### Example input

```json
{
  "tenure": 12,
  "MonthlyCharges": 79.99,
  "TotalCharges": 959.88,
  "gender": "Male",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
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
  "PaymentMethod": "Electronic check"
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

## 🚀 Current Implementation Progress

- Created shared GitHub repository
- Added EDA notebook and data preprocessing workflow
- Trained Logistic Regression and Random Forest models
- Saved trained models and scaler
- Built FastAPI backend for prediction
- Integrated Logistic Regression model into the API
- Added preprocessing and feature mapping for real-time predictions
- Successfully tested the /predict endpoint

---

## 🔜 Future Work

- Tune model hyperparameters
- Experiment with the larger churn dataset
- Improve API input validation and usability
- Add optional frontend interface
- Deploy the application online
