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

## 📁 Project Structure
