import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# Dummy loan data to match the UI feature expectations:
# [Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
X = np.array([
    [1, 1, 0, 1, 0, 5000, 2000, 150, 360, 1, 2],
    [0, 0, 0, 0, 1, 2000, 0, 50, 360, 0, 0],
    [1, 1, 2, 1, 0, 8000, 1000, 250, 360, 1, 1],
    [1, 0, 0, 1, 0, 4000, 0, 100, 360, 1, 2],
    [0, 1, 1, 1, 0, 6000, 3000, 200, 180, 1, 1],
    [1, 1, 0, 0, 1, 3000, 1500, 120, 360, 0, 0]
])
# 1 = Approved, 0 = Rejected
y = np.array([1, 0, 1, 1, 1, 0])

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

os.makedirs('backend/models_store', exist_ok=True)
with open('backend/models_store/loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Fresh model created successfully!")
