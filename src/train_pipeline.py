import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


#Load + clean data
#df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.drop(['customerID'], axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df[df['tenure'] != 0]
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


#Feature engineering
df['AvgCharges'] = df['TotalCharges'] / df['tenure']
df['AvgCharges'] = df['AvgCharges'].fillna(0)

#Split features/target
X = df.drop('Churn', axis=1)
y = df['Churn']


#Define column types
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgCharges']
cat_cols = X.select_dtypes(include='object').columns.tolist()

#Build pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])


#Train pipeline
pipeline.fit(X, y)


#Save pipeline
joblib.dump(pipeline, 'models/churn_pipeline.pkl')

print("Pipeline trained and saved successfully.")