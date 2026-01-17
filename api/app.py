"""
Flask API for flood prediction
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained model (placeholder)
# model = pickle.load(open('models/flood_model.pkl', 'rb'))

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Flood Prediction API",
        "version": "1.0.0",
        "status": "active"
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict flood risk
    
    Expected input:
    {
        "location": {"lat": 19.0760, "lon": 72.8777},
        "current_data": {
            "rainfall_24h": 45.5,
            "river_discharge": 1250.0,
            "soil_moisture": 0.75
        }
    }
    """
    try:
        data = request.get_json()
        
        # Extract features
        rainfall = data['current_data']['rainfall_24h']
        discharge = data['current_data']['river_discharge']
        soil_moisture = data['current_data']['soil_moisture']
        
        # TODO: Make actual prediction with loaded model
        # For demo purposes, simple threshold-based logic
        risk_score = (rainfall * 0.4 + discharge * 0.0003 + soil_moisture * 0.3)
        
        if risk_score > 30:
            risk_level = "HIGH"
        elif risk_score > 20:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        probability = min(risk_score / 40, 1.0)
        
        response = {
            "flood_risk": risk_level,
            "probability": round(probability, 2),
            "predicted_time": "2026-01-20T14:00:00Z",
            "lead_time_hours": 48,
            "confidence_interval": [
                round(probability - 0.05, 2),
                round(probability + 0.05, 2)
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 400

@app.route('/api/historical', methods=['GET'])
def historical():
    """Get historical flood events"""
    # Mock data
    events = [
        {
            "date": "2024-07-15",
            "severity": "HIGH",
            "rainfall_mm": 156.3,
            "affected_area_km2": 45.2
        },
        {
            "date": "2024-08-22",
            "severity": "MEDIUM",
            "rainfall_mm": 98.5,
            "affected_area_km2": 23.1
        }
    ]
    return jsonify(events)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
