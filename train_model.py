import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Preprocessing
train = train.drop("Loan_ID", axis=1)

# Fill missing values
for col in train.columns:
    if train[col].dtype == "object":
        train[col].fillna(train[col].mode()[0], inplace=True)
    else:
        train[col].fillna(train[col].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in train.columns:
    if train[col].dtype == "object":
        train[col] = le.fit_transform(train[col])

# Split data
X = train.drop("Loan_Status", axis=1)
y = train["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")

# Test accuracy
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy: {accuracy}")

# Save test data for drift detection
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)