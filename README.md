# Loan Prediction Backend

This is the backend API for the autonomous loan prediction system with drift detection.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the dataset:
   - Download `train_u6lujuX_CVtuZ9i.csv` from Kaggle (Loan Prediction Dataset)
   - Place it in the project directory

3. Train the model:
   ```
   python train_model.py
   ```
   This will generate `loan_model.pkl`, `X_test.csv`, and `y_test.csv`

4. Run the backend:
   ```
   python backend.py
   ```
   The API will be available at http://localhost:8000

## API Endpoints

- `POST /predict`: Predict loan status
  - Body: JSON with loan application details
- `GET /drift`: Check for data drift
- `GET /accuracy`: Get current model accuracy
- `POST /retrain`: Retrain the model

## Example Prediction Request

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 100,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
```

## Autonomous Features

The backend includes drift detection using KS test and automated retraining when drift is detected.

For production, integrate with logging and monitoring systems.