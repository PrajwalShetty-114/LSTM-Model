# lstm-api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Initialize FastAPI app ---
app = FastAPI(title="LSTM Traffic Forecast API")

# --- 2. Load the trained model and scaler ---
MODEL_PATH = "data/lstm_traffic_model.keras"
SCALER_PATH = "data/lstm_scaler.pkl"
model = None
scaler = None

try:
    model = load_model(MODEL_PATH)
    logger.info(f"Keras model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading Keras model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")

# --- 3. Define the input data structure ---
# This matches what our Node.js server will send
class Coordinates(BaseModel):
    lat: float
    lng: float

class PredictionInput(BaseModel):
    model: str # "lstm"
    coordinates: Coordinates
    selectedDate: str # e.g., "2025-10-31"

# --- 4. Define the prediction endpoint ---
@app.post("/predict/")
async def make_forecast(input_data: PredictionInput):
    logger.info(f"Received forecast request: {input_data.dict()}")

    if model is None or scaler is None:
        logger.error("Model or scaler is not loaded. Cannot make predictions.")
        raise HTTPException(status_code=500, detail="Model or scaler is not available")

    try:
        # --- a. Get Historical Data ---
        # !! IMPORTANT !!
        # Our LSTM model was trained to use the LAST 24 HOURS of data
        # to predict the NEXT hour. To generate a 24-hour forecast,
        # we need to:
        # 1. Get the most recent 24 hours of REAL data for the given coordinates.
        # 2. Predict 1 hour.
        # 3. Use that prediction as input to predict the next hour (and so on).
        
        # --- FOR NOW (DUMMY LOGIC) ---
        # Since we don't have a database of real data, we will create
        # dummy "historical" data to feed the model.
        # This simulates having the last 24 hours of data.
        # We'll just create 24 hours of "medium" traffic (0.5).
        
        # Create dummy historical data
        dummy_history = np.full((24, 1), 0.5) # 24 hours of 0.5
        
        # Scale this dummy history (as the model expects scaled input)
        # Note: We just use .transform, NOT .fit_transform
        scaled_history = scaler.transform(dummy_history)
        
        # Reshape for the model: [1 sample, 24 time_steps, 1 feature]
        current_sequence = scaled_history.reshape(1, 24, 1)

        # --- b. Generate 24-Hour Forecast ---
        forecast_scaled = [] # To store the scaled predictions
        
        for _ in range(24): # Loop 24 times (for 24 hours)
            # Predict the next hour
            next_pred_scaled = model.predict(current_sequence)
            
            # Store the prediction
            forecast_scaled.append(next_pred_scaled[0, 0])
            
            # Update the sequence:
            # Drop the first hour and append the new prediction
            # This is called "walk-forward" prediction
            new_sequence_step = next_pred_scaled.reshape(1, 1, 1)
            current_sequence = np.append(current_sequence[:, 1:, :], new_sequence_step, axis=1)

        # --- c. Inverse Transform (Un-scale) ---
        # We need to un-scale our 24 predictions
        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
        
        # Flatten the array for the JSON response
        forecast_list = [float(round(val, 2)) for val in forecast.flatten()]
        
        # --- d. Format the response ---
        # Send back the data our frontend line chart expects
        response_data = {
            "labels": ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
            "data": forecast_list
        }
        
        logger.info("Successfully generated 24-hour forecast.")
        return response_data

    except Exception as e:
        logger.error(f"Error during forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

# --- 5. Add a simple root endpoint ---
@app.get("/")
def read_root():
    return {"message": "LSTM Traffic Forecast API is running!"}