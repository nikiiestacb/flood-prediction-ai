# flood-prediction-ai
AI-Powered Flood Prediction System using LSTM &amp; CNN | Deep Learning for early warning
# 🌊 AI-Powered Flood Prediction & Early Warning System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning-based flood forecasting system that predicts flood events 24-72 hours in advance using hydrological data, weather forecasts, and satellite imagery. Developed as part of M.Tech research at IIT Bombay.

![Flood Prediction Demo](docs/demo.gif)
*Real-time flood risk prediction dashboard*

---

## 🎯 Problem Statement

Traditional flood warning systems rely on threshold-based alerts which often result in:
- ❌ High false alarm rates (60%+)
- ❌ Missed flood events
- ❌ Insufficient lead time for evacuation

This project develops an **intelligent system** that learns from historical patterns to provide **accurate, timely warnings** with 24-72 hours lead time.

---

## ✨ Key Features

- 🔮 **Multi-horizon Forecasting**: Predicts floods 24, 48, and 72 hours ahead
- 🎯 **High Accuracy**: 92% recall with 87% precision
- 🗺️ **Spatial Analysis**: Covers 500 km² watershed area
- 📊 **Real-time Dashboard**: Interactive risk maps and alerts
- 🔄 **Automated Updates**: Weekly model retraining
- 📱 **Alert System**: SMS/Email notifications to stakeholders

---

## 📊 Performance Metrics

| Metric | Value | Comparison |
|--------|-------|------------|
| **Precision** | 0.87 | +25% vs threshold-based |
| **Recall** | 0.92 | +18% vs threshold-based |
| **F1-Score** | 0.89 | Best in class |
| **Lead Time** | 68 hours avg | 48+ hours for 92% events |
| **False Alarm Rate** | 13% | -35% vs baseline |

---

## 🏗️ Architecture
─────────────────┐
│  Data Sources   │
│  - Weather API  │
│  - USGS Gauges  │
│  - Satellites   │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Data Processing │
│  - Cleaning     │
│  - Feature Eng. │
└────────┬────────┘
│
▼
┌─────────────────┐
│  ML Pipeline    │
│  - XGBoost      │
│  - LSTM         │
│  - CNN-LSTM     │
└────────┬────────┘
│
▼
┌─────────────────┐
│  Ensemble Model │
│  - Prediction   │
│  - Uncertainty  │
└────────┬────────┘
│
▼
┌─────────────────┐
│  API & Dashboard│
│  - Flask API    │
│  - Web Dashboard│
│  - Alerts       │
└─────────────────┘

---

## 🛠️ Tech Stack

**Machine Learning:**
- **XGBoost**: Gradient boosting for 24-hour predictions
- **LSTM**: Long Short-Term Memory for sequential patterns
- **CNN-LSTM**: Hybrid model for spatial-temporal features
- **Ensemble**: Weighted average of top 3 models

**Data Processing:**
- Pandas, NumPy, GeoPandas
- Scikit-learn for preprocessing
- Rasterio for satellite imagery

**Deployment:**
- Flask for REST API
- Docker for containerization
- AWS EC2 for hosting
- PostgreSQL + PostGIS for data storage

**Visualization:**
- Plotly for interactive charts
- Folium for maps
- Matplotlib/Seaborn for static plots

---

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- (Optional) Docker for containerized deployment

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/flood-prediction-ai.git
cd flood-prediction-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

5. **Download sample data** (Optional - for testing)
```bash
python src/data_processing/download_sample_data.py
```

### Running the Project

**1. Data Processing**
```bash
python src/data_processing/process_data.py
```

**2. Feature Engineering**
```bash
python src/features/build_features.py
```

**3. Train Models**
```bash
python src/models/train_model.py
```

**4. Run API**
```bash
python api/app.py
# API will be available at http://localhost:5000
```

**5. View Dashboard**
```bash
# Open browser and go to http://localhost:5000/dashboard
```

---

## 📓 Notebooks

Explore the analysis in interactive Jupyter notebooks:

1. **Data Exploration** - Understanding the dataset, patterns, and correlations
2. **Feature Engineering** - Creating predictive features from raw data
3. **Model Training** - Training and comparing different ML models
4. **Model Evaluation** - Detailed performance analysis
5. **Visualization** - Creating maps and charts for insights

To run notebooks:
```bash
jupyter notebook notebooks/
```

---

## 🔬 Methodology

### Data Sources

1. **Weather Data**: NOAA API - Temperature, precipitation, humidity
2. **Stream Gauges**: USGS - River discharge, water levels
3. **Satellite Imagery**: Sentinel-2 - Land use, water extent
4. **Topography**: SRTM DEM - Elevation, slope, drainage

### Feature Engineering (45+ features)

**Temporal Features:**
- Antecedent precipitation index (API)
- Cumulative rainfall (3, 7, 14, 30 days)
- River discharge rate of change
- Seasonal indicators
- Lagged variables (1-7 days)

**Spatial Features:**
- Drainage area
- Average slope
- Topographic wetness index
- Distance to river network
- Land cover types
- Soil saturation index

### Model Training

**Baseline:** Threshold-based warning system

**Model 1: XGBoost**
- Best for 24-hour ahead predictions
- Feature importance analysis
- Hyperparameter tuning with Optuna

**Model 2: LSTM**
- Multi-step forecasting (24, 48, 72 hours)
- Sequence length: 30 days
- 3-layer architecture with dropout

**Model 3: CNN-LSTM**
- Combines spatial (CNN) and temporal (LSTM) patterns
- Processes satellite imagery + time series
- Best for long-range predictions

**Ensemble:**
- Weighted average based on validation performance
- Uncertainty quantification

### Evaluation

- **Time-based split**: Train (70%), Validation (15%), Test (15%)
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Business Metrics**: Lead time, false alarm rate, missed events
- **Comparison**: Against operational flood warning systems

---

## 📈 Results

### Model Performance

| Model | Precision | Recall | F1-Score | Lead Time (hrs) |
|-------|-----------|--------|----------|-----------------|
| Threshold-based | 0.62 | 0.74 | 0.68 | 24 |
| XGBoost | 0.85 | 0.89 | 0.87 | 32 |
| LSTM | 0.83 | 0.91 | 0.87 | 52 |
| CNN-LSTM | 0.84 | 0.90 | 0.87 | 58 |
| **Ensemble** | **0.87** | **0.92** | **0.89** | **68** |

### Business Impact

- ✅ **Successfully predicted 92% of flood events** with >48 hours lead time
- ✅ **Reduced false alarm rate by 35%** compared to threshold system


---

## 🌐 API Documentation

### Predict Endpoint
```bash
POST /api/predict
```

**Request Body:**
```json
{
  "location": {"lat": 19.0760, "lon": 72.8777},
  "current_data": {
    "rainfall_24h": 45.5,
    "river_discharge": 1250.0,
    "soil_moisture": 0.75
  }
}
```

**Response:**
```json
{
  "flood_risk": "HIGH",
  "probability": 0.87,
  "predicted_time": "2026-01-20T14:00:00Z",
  "lead_time_hours": 68,
  "confidence_interval": [0.82, 0.92]
}
```

See [API Documentation](docs/api_docs.md) for full details.

---

## 🐳 Docker Deployment

Build and run with Docker:
```bash
# Build image
docker build -t flood-prediction .

# Run container
docker run -p 5000:5000 flood-prediction
```

Or use Docker Compose:
```bash
docker-compose up
```

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

---

## 📊 Visualizations

### Flood Risk Map
![Risk Map](docs/images/risk_map.png)

### Model Performance
![Performance](docs/images/model_performance.png)

### Feature Importance
![Features](docs/images/feature_importance.png)

---

## 🎓 Academic Context

This project was developed as part of M.Tech research at **IIT Bombay** in the Department of Water Resources Engineering.

**Title**: "Machine Learning Approaches for Flood Prediction and Early Warning Systems"






