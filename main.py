
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# 1. Load the trained model and feature columns
model = joblib.load("model.pkl")

with open("feature_columns.json", "r") as file:
    feature_columns = json.load(file)


# 2. Create FastAPI app
app = FastAPI()

# Enable CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Define the request body
class PredictionInput(BaseModel):
    event_type: str
    ticket_price: int
    days_before_event: int
    reminder_sent: int
    reminder_opened: int
    prior_attendance: int
    cancelled: int


# 4. Create the prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    # Convert the input into a DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # One-hot encode event_type
    input_data = pd.get_dummies(
        input_data,
        columns=["event_type"],
        dtype=int
    )

    # Align columns with the columns used during model training
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # 5. Get attendance probability
    probability = model.predict_proba(input_data)[0][1]

    # 6. Determine no-show risk
    if probability > 0.7:
        no_show_risk = "Low"
    elif probability > 0.4:
        no_show_risk = "Medium"
    else:
        no_show_risk = "High"

    # Get feature importances
    importances = model.feature_importances_

    # Sort features by importance and get top 3
    top_indices = importances.argsort()[::-1][:3]
    top_reasons = [
        feature_columns[i]
        for i in top_indices
    ]

    return {
        "attendance_probability": float(probability),
        "no_show_risk": no_show_risk,
        "top_reasons": top_reasons
    }


# 7. Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
