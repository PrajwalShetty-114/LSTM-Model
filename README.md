# LSTM Traffic Forecast API 🧠📈

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?style=for-the-badge&logo=tensorflow)

This repository contains the backend microservice for the **LSTM (Future Forecaster)** model, a core component of the "Traffic Flow Prediction" project.

The purpose of this API is to receive a location and a future date, load a pre-trained Keras LSTM model, and return a 24-hour time-series forecast for traffic congestion.

## 🚀 Role in Project Architecture

This service is not intended to be called directly by the public. It is a backend microservice designed to be called by the main **Node.js API Gateway**.

**Data Flow:**
`Frontend` ➔ `Node.js Gateway` ➔ `This LSTM API`

## 📦 API Endpoints

### 1. Health Check

* **Endpoint:** `GET /`
* **Description:** A simple endpoint to confirm the API is running.
* **Response (200 OK):**
    ```json
    { "message": "LSTM Traffic Forecast API is running!" }
    ```

### 2. Make Forecast

* **Endpoint:** `POST /predict/`
* **Description:** Receives location and date, then generates a 24-hour forecast.
* **Request Body:**
    ```json
    {
      "model": "lstm",
      "coordinates": { "lat": 12.9716, "lng": 77.5946 },
      "selectedDate": "2025-11-20"
    }
    ```
* **Success Response (200 OK):**
    * Returns the labels (`labels`) and predicted values (`data`) our frontend line chart expects.
    ```json
    {
      "labels": ["00:00", "01:00", "02:00", ... , "23:00"],
      "data": [0.1, 0.1, 0.2, ... , 0.3, 0.2]
    }
    ```
* **Error Response (500):**
    * Returns an error if the model or scaler files are not loaded.
    ```json
    { "detail": "Model or scaler is not available" }
    ```

## 🛠️ Setup and Installation

Here are the step-by-step commands to set up and run this server locally.

### 1. Clone the Repository

```bash
# Clone this repository
git clone https://github.com/PrajwalShetty-114/LSTM-Model.git
cd traffic-prediction-lstm-api
```
### 2. Set Up Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install Git LFS
This project uses Git Large File Storage (LFS) for the `.keras` and `.pkl` model files.

```bash
# Install Git LFS (if not already installed)
git lfs install

# Pull the large files from LFS storage
git lfs pull

```

**Part 2: Install Dependencies**
```markdown

All required packages are listed in `requirements.txt`.

# Install all required Python libraries
pip install -r requirements.txt
```
**Part 3: Run the Server**
```markdown

This API is configured to run on **port 8002** to avoid conflicts with other project services.

# Run the FastAPI server with auto-reload
uvicorn main:app --reload --port 8002
```
The server will be available at http://127.0.0.1:8002.

**Part 4: Model Details**

* **Model:** `data/lstm_traffic_model.keras`
    * A Keras/TensorFlow Sequential model trained in Google Colab.
* **Scaler:** `data/lstm_scaler.pkl`
    * An `sklearn.preprocessing.MinMaxScaler` object.
    * This is **crucial** as it's used to scale the input data (0-1) before prediction and inverse-transform the output (0-1) back into a real traffic value.

