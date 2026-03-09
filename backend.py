from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import uvicorn

app = FastAPI(title="Loan Prediction API", description="Autonomous ML Model Monitoring & Drift Management API")

# Load model
try:
    model = joblib.load("loan_model.pkl")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found")

# Load test data for drift detection
try:
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")
except FileNotFoundError:
    X_test = None
    y_test = None

# Categorical mappings
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
education_map = {'Graduate': 0, 'Not Graduate': 1}
self_employed_map = {'Yes': 1, 'No': 0}
property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
credit_history_map = {1.0: 1, 0.0: 0}

class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.post("/predict")
def predict_loan(application: LoanApplication):
    try:
        input_data = {
            'Gender': gender_map[application.Gender],
            'Married': married_map[application.Married],
            'Dependents': dependents_map[application.Dependents],
            'Education': education_map[application.Education],
            'Self_Employed': self_employed_map[application.Self_Employed],
            'ApplicantIncome': application.ApplicantIncome,
            'CoapplicantIncome': application.CoapplicantIncome,
            'LoanAmount': application.LoanAmount,
            'Loan_Amount_Term': application.Loan_Amount_Term,
            'Credit_History': credit_history_map[application.Credit_History],
            'Property_Area': property_area_map[application.Property_Area]
        }
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        result = "Approved" if prediction == 1 else "Rejected"
        return {"prediction": result, "status_code": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drift")
def check_drift():
    if X_test is None:
        raise HTTPException(status_code=500, detail="Test data not available")
    # Simulate production data (in real scenario, collect from logs)
    production_data = X_test.sample(min(100, len(X_test)), random_state=42)
    drift_report = {}
    for column in X_test.columns:
        stat, p_value = ks_2samp(X_test[column], production_data[column])
        drift_report[column] = float(p_value)
    return {"drift_report": drift_report}

@app.get("/accuracy")
def get_accuracy():
    if y_test is None:
        raise HTTPException(status_code=500, detail="Test data not available")
    preds = model.predict(X_test)
    accuracy = float(np.mean(preds == y_test.values.flatten()))
    return {"accuracy": accuracy}

@app.post("/retrain")
def retrain_model():
    # This would require the original training data
    # For demo, we'll just reload or retrain with available data
    # In production, load full dataset
    try:
        # Assuming train.csv is available
        train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
        # Preprocessing as in train_model.py
        train = train.drop("Loan_ID", axis=1)
        for col in train.columns:
            if train[col].dtype == "object":
                train[col].fillna(train[col].mode()[0], inplace=True)
            else:
                train[col].fillna(train[col].median(), inplace=True)
        le = LabelEncoder()
        for col in train.columns:
            if train[col].dtype == "object":
                train[col] = le.fit_transform(train[col])
        X = train.drop("Loan_Status", axis=1)
        y = train["Loan_Status"]
        new_model = RandomForestClassifier()
        new_model.fit(X, y)
        joblib.dump(new_model, "loan_model.pkl")
        global model
        model = new_model
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)