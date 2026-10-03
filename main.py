
import json
import joblib
import pandas as pd

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# 1. Load the trained model and feature columns
model = joblib.load("model.pkl")

with open("feature_columns.json", "r") as file:
    feature_columns = json.load(file)


# 2. Create the FastAPI application
app = FastAPI()

# Enable CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Define the input model with the exact fields and order
class AttendanceInput(BaseModel):
    registration_id: str
    event_id: str
    event_category: str
    ticket_price_taka: int
    days_before_event_registered: int
    payment_delay_hours: Optional[float] = None
    event_day_of_week: int
    event_start_hour: int
    location_type: str
    reminder_status: str
    prior_attendance_count: int
    is_cancelled: bool


# 4. Create the attendance prediction endpoint
@app.post("/predict/forecast")
def predict_forecast(data: AttendanceInput):

    # Convert the request to a dictionary
    input_data = data.model_dump()

    # Preserve identifiers for the response
    registration_id = input_data["registration_id"]
    event_id = input_data["event_id"]

    # Convert is_cancelled to integer
    input_data["is_cancelled"] = (
        1 if input_data["is_cancelled"] else 0
    )

    # Replace missing payment delay with -1
    if input_data["payment_delay_hours"] is None:
        input_data["payment_delay_hours"] = -1

    # Build a single-row DataFrame containing ONLY
    # the 10 predictive features in the specified order
    predictive_features = [
        "event_category",
        "ticket_price_taka",
        "days_before_event_registered",
        "payment_delay_hours",
        "event_day_of_week",
        "event_start_hour",
        "location_type",
        "reminder_status",
        "prior_attendance_count",
        "is_cancelled"
    ]

    row = {
        feature: input_data[feature]
        for feature in predictive_features
    }

    input_df = pd.DataFrame(
        [row],
        columns=predictive_features
    )

    # Apply one-hot encoding to categorical features
    categorical_columns = [
        "event_category",
        "location_type",
        "reminder_status"
    ]

    input_df = pd.get_dummies(
        input_df,
        columns=categorical_columns,
        dtype=int
    )

    # Align columns with the training feature columns
    # Missing columns are filled with 0; extra columns are dropped
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Get attendance probability for the positive class (checked_in = 1)
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # Determine no-show risk
    if probability > 0.7:
        no_show_risk = "Low"
    elif probability > 0.4:
        no_show_risk = "Medium"
    else:
        no_show_risk = "High"

    # Get the top 3 feature names by importance
    importances = model.feature_importances_

    top_indices = importances.argsort()[::-1][:3]

    top_reasons = [
        feature_columns[index]
        for index in top_indices
    ]

    # Return the prediction as JSON
    return {
        "registration_id": registration_id,
        "event_id": event_id,
        "attendance_probability": probability,
        "no_show_risk": no_show_risk,
        "top_reasons": top_reasons
    }


# 5. Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
