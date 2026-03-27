# Customer-Churn-Prediction

👥 Team Members
- Trista Chen — trista.chen@sjsu.edu
- Sonal Rana — sonal.rana@sjsu.edu

---

## 🧩 Problem Statement

Customer churn is a major challenge for telecom companies, as losing existing customers leads to significant revenue loss and increased costs to acquire new customers.

The goal of this project is to build a machine learning model that predicts whether a customer is likely to churn based on their demographic information, service usage, and account details. This prediction system can help businesses take proactive steps to improve customer retention.

---

## 📂 Dataset

We are using a telecom customer dataset from Kaggle:
- Source: Customer Churn Prediction Dataset
- Link: https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction

Features include:
- Customer demographics (gender, senior citizen, partner, dependents)
- Account information (tenure, contract type, billing method)
- Services used (internet service, streaming, tech support, etc.)
- Financial data (monthly charges, total charges)

Target Variable:
- Churn (Yes/No)

---

## ⚙️ Planned Approach

### 1. Data Preprocessing
- Handle missing values
- Encode categorical variables
- Scale numerical features

### 2. Exploratory Data Analysis (EDA)
- Analyze patterns affecting churn (e.g., tenure, contract type)
- Visualize relationships between features and churn

### 3. Model Development
- Train classification models such as:
  - Logistic Regression
  - Random Forest
- Evaluate using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

### 4. Deployment (FastAPI)
- Build an API that:
  - Accepts customer input
  - Returns churn probability
  - Classifies risk (Low / Medium / High)
  - Provides retention recommendations

---

## 🗂️ Project Structure

```text
customer-churn/
│
├── data/              # Dataset files
├── notebooks/         # EDA and experiments
├── src/               # Data processing & model training
├── models/            # Saved ML models
├── app/               # FastAPI backend
├── requirements.txt   # Dependencies
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- FastAPI
- Matplotlib
- Pandas / NumPy

---

## ▶️ How to Run (Basic)

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app.main:app --reload
```

---

## 📌 Future Work

- Improve model performance with hyperparameter tuning
- Add more advanced models (e.g., XGBoost)
- Build a simple frontend interface
- Deploy the application online
