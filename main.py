# lstm-api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LSTM Traffic Forecast API")

# --- 1. Load Model & Scaler ---
MODEL_PATH = "data/lstm_traffic_model.keras"
SCALER_PATH = "data/lstm_scaler.pkl"
model = None
scaler = None

try:
    model = load_model(MODEL_PATH)
    logger.info(f"✅ Keras model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading Keras model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"✅ Scaler loaded from {SCALER_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading scaler: {e}")

# --- 2. Input Models ---
class Coordinates(BaseModel):
    lat: float
    lng: float

class PredictionInput(BaseModel):
    model: str
    coordinates: Coordinates
    selectedDate: str

# --- 3. Helper: Generate Unique History per Location ---
def generate_realistic_history(lat, lng):
    """
    Generates a unique, realistic 24-hour traffic pattern for a specific location.
    This replaces the 'dummy flat line' with a curve that respects rush hours
    but varies based on the coordinate seed.
    """
    # 1. Create a deterministic seed from coordinates
    # This ensures the same location always gives the same 'history'
    seed = int((lat + lng) * 100000)
    np.random.seed(seed)

    # 2. Base Traffic Level (Random per location)
    # Some roads are busier (30k), some quieter (5k)
    base_volume = np.random.randint(5000, 30000)

    # 3. Generate 24 hours of data
    history = []
    for hour in range(24):
        # A simple curve: Low at night, peaks at 9AM (9) and 6PM (18)
        # We use Sin/Cos math to shape the day
        morning_peak = np.exp(-0.1 * (hour - 9)**2)  # Peak around 9
        evening_peak = np.exp(-0.1 * (hour - 18)**2) # Peak around 18
        night_lull = -0.5 * np.exp(-0.1 * (hour - 3)**2) # Dip around 3 AM
        
        # Combine factors
        activity_factor = 1.0 + morning_peak + evening_peak + night_lull
        
        # Add random noise specific to this hour and location
        noise = np.random.normal(0, 0.1) 
        
        # Calculate volume for this hour
        hourly_vol = base_volume * (activity_factor + noise)
        
        # Ensure non-negative
        hourly_vol = max(100, hourly_vol)
        history.append(hourly_vol)

    return np.array(history).reshape(24, 1)

# --- 4. Prediction Endpoint ---
@app.post("/predict/")
async def make_forecast(input_data: PredictionInput):
    logger.info(f"Received forecast request for: {input_data.coordinates}")

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler is not available")

    try:
        # A. Generate Location-Specific History
        # Instead of a flat line, we generate a unique curve for this lat/lng
        raw_history = generate_realistic_history(
            input_data.coordinates.lat, 
            input_data.coordinates.lng
        )
        
        # B. Scale the History
        # The model expects values between 0 and 1
        scaled_history = scaler.transform(raw_history)
        
        # Reshape for LSTM: [1 sample, 24 time_steps, 1 feature]
        current_sequence = scaled_history.reshape(1, 24, 1)

        # C. Predict the NEXT 24 Hours (Walk-Forward)
        forecast_scaled = []
        
        for _ in range(24):
            # 1. Predict next step
            next_step_pred = model.predict(current_sequence, verbose=0) # verbose=0 hides logs
            val = next_step_pred[0, 0]
            forecast_scaled.append(val)
            
            # 2. Update sequence (Drop first, add new prediction)
            new_step = np.array([[[val]]])
            current_sequence = np.append(current_sequence[:, 1:, :], new_step, axis=1)

        # D. Inverse Transform (Scale Back to Real Numbers)
        forecast_array = np.array(forecast_scaled).reshape(-1, 1)
        forecast_real = scaler.inverse_transform(forecast_array)
        
        # Flatten to simple list and ensure python floats
        forecast_list = [float(round(val, 0)) for val in forecast_real.flatten()]

        # E. Response
        response_data = {
            "labels": [f"{h:02d}:00" for h in range(24)],
            "data": forecast_list
        }
        
        return response_data

    except Exception as e:
        logger.error(f"Error during forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

@app.get("/")
def read_root():
    return {"message": "LSTM Traffic Forecast API is running!"}