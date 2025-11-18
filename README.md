# 🌍 Air Quality Prediction - Skåne, Sweden

Serverless ML system predicting PM2.5 air quality 7 days ahead for cities in Skåne using XGBoost and lagged features.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([your-app-url](https://streamlit.io/))
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

🔗 **[Live Dashboard](https://air-quality-skane.streamlit.app/)**

---

## Overview

Automated pipeline that collects daily air quality (PM2.5) and weather data, trains an XGBoost using Bayersian hyperparameter optimization model with **lagged features** (past 3 days), and generates 7 day forecasts.

**Locations:** Everöd, Ludvigsborg, Eslöv, Laröd, Asmundtorp, Dösjebro, Skillinge (Skåne)

---

## Quick Start
```bash
# Clone and install
git clone https://github.com/yourusername/air-quality-predictor.git
cd air-quality-predictor
uv sync

# Setup .env
cp .env.example .env
# Add HOPSWORKS_API_KEY and AQICN_API_KEY. Can delete the OpenAI key

# Run pipelines
uv run python backfill-feature-pipeline.py
uv run python training-pipeline.py
uv run python batch-inference-pipeline.py
uv run streamlit run dashboard.py
```

---

## Structure
```
├── backfill.py                   # Historical data loading
├── feature-daily_pipeline.py     # Daily feature collection
├── training-pipeline.py          # Model training with lagged features
├── batch-inference-pipeline.py   # Generate predictions
├── dashboard.py                  # Streamlit dashboard
├── util.py                       # Helper functions
├── locations.json                # Skåne stations
└── .github/workflows/
    └── air-quality-daily.yml     # Automated daily runs (06:11 UTC)
```

---

## Feature Engineering

### **PM2.5 Lag & Rolling Features**

| Feature       | Description        |
| ------------- | ------------------ |
| `pm25_lag_1`  | PM2.5 yesterday    |
| `pm25_lag_2`  | 2 days ago         |
| `pm25_lag_3`  | 3 days ago         |
| `pm25_roll_3` | 3-day rolling mean |

**These are the strongest predictors** due to temporal autocorrelation in pollution levels.

### **Weather-Based Features**

From Open-Meteo:

* `temperature_2m_mean`
* `precipitation_sum`
* `wind_speed_10m_max`
* `wind_direction_10m_dominant`

Plus 1-day lags:

* `temperature_2m_mean_lag_1`
* `precipitation_sum_lag_1`
* `wind_speed_10m_max_lag_1`
* `wind_direction_10m_dominant_lag_1`

And rolling features:

* `weather_temp_roll_3`
* `weather_wind_roll_3`

### **Calendar Features**

* `day_of_week`

### ** Spatial Features**

* `Latitude` & `longitude` of each station
---

## Hyperparameter Optimization 

The model uses **Optuna** to perform Bayesian hyperparameter search over:

* `max_depth`
* `learning_rate`
* `n_estimators`
* `subsample`
* `colsample_bytree`
* `gamma`
* `min_child_weight`

Search objective:

```python
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "tree_method": "hist",
        "n_jobs": -1,
    }
```

Optuna runs 30 trials and selects the best configuration using validation MSE.

---

## Model Performance

| Feature Set | R² Score | RMSE |
|-------------|----------|------|
| Weather only | -5.323 | 39.854 μg/m³|
| **+ Lagged** | 0.9769 | 0.3822 μg/m³|

**Why lagged features help:** PM2.5 levels are strongly autocorrelated, so adding lagged values and rolling averages gives the model direct access to the most important predictors. Lagged weather features capture delayed meteorological effects, and spatial features account for consistent station-to-station differences. Together, these signals explain most short-term PM2.5 variability, raising R² from negative values to around 0.93. The hyperparameter tuning we did accounts for the $R^2$ reaching the last bit to 0.97

---

## Automated Pipeline

**Runs daily at 06:11 UTC via GitHub Actions:**
1. Fetch yesterday's air quality + weather
2. Get 7 day weather forecasts
3. Generate predictions
4. Update dashboard

---

## Data Sources & Stack

**Data:**
- Air Quality: [AQICN](https://aqicn.org/)
- Weather: [Open-Meteo](https://open-meteo.com/)

**Infrastructure:**
- Feature Store: [Hopsworks](https://www.hopsworks.ai/)
- Model: XGBoost with lagged features
- Orchestration: GitHub Actions
- Dashboard: Streamlit Cloud

---

## Academic Context

**Course:** ID2223 Scalable Machine Learning and Deep Learning  
**Institution:** KTH Royal Institute of Technology  
**Based on:** [Serverless ML Course](https://github.com/featurestorebook/mlfs-book) by Prof. Jim Dowling

---
