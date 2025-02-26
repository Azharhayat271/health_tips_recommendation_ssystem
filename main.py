from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved models and encoders
activity_model = joblib.load("activity_model.pkl")
health_model = joblib.load("health_model.pkl")
activity_encoder = joblib.load("activity_encoder.pkl")
health_tip_encoder = joblib.load("health_tip_encoder.pkl")

# Define input data model using Pydantic
class InputData(BaseModel):
    glucose_level: float
    age: int
    bmi: float
    meal_code: int

# Initialize FastAPI app
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to numpy array
        input_data = np.array([[data.glucose_level, data.age, data.bmi, data.meal_code]])

        # Make predictions
        activity_prediction = activity_model.predict(input_data)[0]
        health_prediction = health_model.predict(input_data)[0]

        # Decode predictions using the LabelEncoders
        activity_label = activity_encoder.inverse_transform([activity_prediction])[0]
        health_tip_label = health_tip_encoder.inverse_transform([health_prediction])[0]

        # Return decoded predictions
        return {
            "activity_recommendation": activity_label,
            "health_tip_recommendation": health_tip_label
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Health Recommendation API!"}
