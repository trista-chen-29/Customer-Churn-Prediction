# 📊 Customer Churn Prediction (In Progress)

## 👥 Team Members
- Trista Chen — trista.chen@sjsu.edu  
- Sonal Rana — sonal.rana@sjsu.edu  

---

## 🧩 Problem Statement
Customer churn is a major challenge for telecom companies, as losing existing customers leads to significant revenue loss and higher costs to acquire new customers.

This project aims to build a machine learning model that predicts whether a customer is likely to churn based on demographic information, service usage, and account details. The system will support data-driven decisions to improve customer retention.

---

## 📂 Dataset

We are using a telecom customer dataset from Kaggle:

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


> ⚠️ Note: The dataset is not stored in this repository. Please download it from the link above and place it in the `data/` folder.

---

## ⚙️ Planned Approach

### 1. Data Preprocessing
- Handle missing values  
- Encode categorical variables  
- Scale numerical features  

### 2. Exploratory Data Analysis (EDA)
- Analyze patterns affecting churn (e.g., tenure, contract type)  
- Visualize feature relationships  

### 3. Model Development
- Train classification models:
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
Customer-Churn-Prediction/
│
├── app/                # FastAPI backend
├── src/                # ML scripts (preprocessing, training)
├── models/             # Saved models (.pkl)
├── data/               # Dataset (not tracked in Git)
├── notebooks/          # EDA notebooks
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- FastAPI
- Pandas / NumPy
- Matplotlib

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app.main:app --reload
```

Then open:
http://127.0.0.1:8000/docs

---

## 🚀 Current Implementation Progress

- Created shared GitHub repository
- Set up FastAPI backend structure
- Implemented /predict API endpoint
- Added input schema and response logic
- Prepared project structure for ML pipeline

---

## 🔜 Next Steps

- Train baseline machine learning model
- Save trained model (.pkl)
- Integrate model into FastAPI backend

---

## 📌 Future Work

- Improve model performance with tuning
- Experiment with additional datasets
- Build optional frontend interface
- Deploy application