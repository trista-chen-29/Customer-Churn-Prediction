import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
#Load + clean data
#df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.drop(['customerID'], axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df[df['tenure'] != 0]
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


#Feature engineering
df['AvgCharges'] = df['TotalCharges'] / df['tenure']
df['AvgCharges'] = df['AvgCharges'].fillna(0)

#Split features/target
X = df.drop('Churn', axis=1)
y = df['Churn']



#Define column types
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

#Build pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000,class_weight='balanced'))
])


# Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# train
pipeline.fit(X_train, y_train)
models = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42) }
#evaluate prediction
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
results = {}
for name, model in models.items():
    print(f"\n{'/'*40}")
    print(f"Model: {name}")
    print(f"{'/'*40}")
    
    # Create pipeline for each model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = cm
    
    # Print metrics
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

#Save pipeline
joblib.dump(pipeline, 'models/churn_pipeline.pkl')
print("Pipeline trained and saved successfully.")