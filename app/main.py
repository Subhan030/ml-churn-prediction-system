from fastapi import FastAPI
from app.schemas import CustomerInput
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability for telecom customers",
    version="1.0"
)

# Load model ONCE
MODEL_PATH = os.path.join("models", "gb_churn_model.joblib")
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.3
@app.post("/predict")
def predict_churn(customer: CustomerInput):
    # Convert input to DataFrame (1-row)
    data = pd.DataFrame([customer.dict()])

    # Predict probability
    churn_proba = model.predict_proba(data)[:, 1][0]

    # Apply threshold
    churn_pred = int(churn_proba >= THRESHOLD)

    return {
        "churn_probability": round(float(churn_proba), 4),
        "churn_prediction": churn_pred,
        "threshold": THRESHOLD
    }


@app.get("/")
def health_check():
    return {"status": "ok"}
