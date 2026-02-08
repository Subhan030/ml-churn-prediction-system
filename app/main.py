from fastapi import FastAPI

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability for telecom customers",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "ok"}
