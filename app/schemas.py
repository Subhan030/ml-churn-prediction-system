from pydantic import BaseModel, Field
from typing import Literal
from fastapi import FastAPI, HTTPException

import joblib
import pandas as pd
import os

class CustomerInput(BaseModel):
    tenure: int = Field(..., ge=0, le=100)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    InternetService: Literal["DSL", "Fiber optic", "No"]

    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PaperlessBilling: Literal["Yes", "No"]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability for telecom customers",
    version="1.0"
)

MODEL_PATH = os.path.join("models", "gb_churn_model.joblib")
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.3


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    try:
        data = pd.DataFrame([customer.dict()])
        churn_proba = model.predict_proba(data)[:, 1][0]
        churn_pred = int(churn_proba >= THRESHOLD)

        return {
            "churn_probability": round(float(churn_proba), 4),
            "churn_prediction": churn_pred,
            "threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
