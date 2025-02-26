from fastapi import FastAPI
import joblib
import numpy as np

# Load trained models
activity_model = joblib.load("activity_model.pkl")
health_model = joblib.load("health_model.pkl")

# Load encoders
activity_encoder = joblib.load("activity_encoder.pkl")
health_tip_encoder = joblib.load("health_tip_encoder.pkl")
meal_encoder = joblib.load("meal_encoder.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Health Recommendation API!"}

@app.post("/predict/")
def predict(glucose: float, age: int, bmi: float, meal_status: str):
    # Encode meal status
    meal_code = meal_encoder.transform([meal_status])[0]
    
    # Prepare input data
    input_data = np.array([[glucose, age, bmi, meal_code]])
    
    # Make predictions
    activity_prediction = activity_model.predict(input_data)[0]
    health_prediction = health_model.predict(input_data)[0]
    
    # Decode predictions
    predicted_activity = activity_encoder.inverse_transform([activity_prediction])[0]
    predicted_health_tip = health_tip_encoder.inverse_transform([health_prediction])[0]
    
    return {
        "Predicted Activity": predicted_activity,
        "Predicted Health Tip": predicted_health_tip
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)