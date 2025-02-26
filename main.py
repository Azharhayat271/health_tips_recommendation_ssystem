from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved models
activity_model = joblib.load("activity_model.pkl")
health_model = joblib.load("health_model.pkl")

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

        # Return predictions
        return {
            "activity_prediction": int(activity_prediction),
            "health_prediction": int(health_prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Health Recommendation API!"}
