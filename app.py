from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import MODELS_DIR


class LoanApplication(BaseModel):
    Applicant_Income: float = Field(..., ge=0)
    Coapplicant_Income: float = Field(..., ge=0)
    Employment_Status: Literal["Salaried", "Self-Employed", "Business"]
    Age: int = Field(..., ge=18, le=80)
    Marital_Status: Literal["Married", "Single"]
    Dependents: int = Field(..., ge=0, le=10)
    Credit_Score: int = Field(..., ge=300, le=900)
    Existing_Loans: int = Field(..., ge=0, le=20)
    DTI_Ratio: float = Field(..., ge=0, le=100)
    Savings: float = Field(..., ge=0)
    Collateral_Value: float = Field(..., ge=0)
    Loan_Amount: float = Field(..., ge=1000)
    Loan_Term: int = Field(..., ge=6, le=480)
    Loan_Purpose: Literal["Home", "Education", "Personal", "Business"]
    Property_Area: Literal["Urban", "Semi-Urban", "Rural"]
    Education_Level: Literal["Graduate", "Postgraduate", "Undergraduate"]
    Gender: Literal["Male", "Female"]
    Employer_Category: Literal["Govt", "Private", "Self"]


app = FastAPI(title="CreditWise Loan Approval API", version="1.0.0")
MODEL_PATH = MODELS_DIR / "best_model.pkl"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(application: LoanApplication):
    if not MODEL_PATH.exists():
        return {
            "error": "Model not found. Run preprocessing, baseline training, and tuning first."
        }

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([application.model_dump()])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    decision = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": prediction,
        "decision": decision,
        "approval_probability": round(probability, 4),
    }
