import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    recall_score
)

import seaborn as sns
import matplotlib.pyplot as plt

# Load + clean data

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.drop(['customerID'], axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df[df['tenure'] != 0]
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Feature engineering

df['AvgCharges'] = df['TotalCharges'] / df['tenure']
df['AvgCharges'] = df['AvgCharges'].fillna(0)

# Split features/target

X = df.drop('Churn', axis=1)
y = df['Churn']

# Define column types

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

# Preprocessing pipeline

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

# Evaluation setup

results = {}

best_model = None
best_pipeline = None
best_score = 0
best_recall = 0

# Train + evaluate models
for name, model in models.items():

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Store results
    results[name] = {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall
    }

    # Track best model
    if f1 > best_score:
        best_score = f1
        best_recall = recall
        best_model = name
        best_pipeline = pipeline

    # Print metrics
    print("Confusion Matrix:\n")
    print(cm)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall Score: {recall:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn']
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()

# Print best model

print("\n" + "="*50)
print(f"Best Performing Model: {best_model}")
print(f"Best F1 Score: {best_score:.4f}")
print(f"Recall Score: {best_recall:.4f}")
print("="*50)

# Save best pipeline

joblib.dump(best_pipeline, 'models/churn_pipeline.pkl')

print("\nBest pipeline trained and saved successfully.")