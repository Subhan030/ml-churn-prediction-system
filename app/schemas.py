from pydantic import BaseModel, Field
from typing import Literal

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
