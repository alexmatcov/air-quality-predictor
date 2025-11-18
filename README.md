# 🌍 Air Quality Prediction - Skåne, Sweden

Serverless ML system predicting PM2.5 air quality 7 days ahead for cities in Skåne using XGBoost and lagged features.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-app-url)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

🔗 **[Live Dashboard](https://air-quality-skane.streamlit.app/)**

---

## 📋 Overview

Automated pipeline that collects daily air quality (PM2.5) and weather data, trains an XGBoost model with **lagged features** (past 3 days), and generates 7 day forecasts.

**Locations:** Everöd, Ludvigsborg, Eslöv, Laröd, Asmundtorp, Dösjebro, Skillinge (Skåne)

---

## 🚀 Quick Start
```bash
# Clone and install
git clone https://github.com/yourusername/air-quality-predictor.git
cd air-quality-predictor
uv sync

# Setup .env
cp .env.example .env
# Add HOPSWORKS_API_KEY and AQICN_API_KEY

# Run pipelines
uv run python backfill.py
uv run python training_pipeline.py
uv run streamlit run dashboard.py
```

---

## 📁 Structure
```
├── backfill.py                   # Historical data loading
├── feature_daily_pipeline.py     # Daily feature collection
├── training_pipeline.py          # Model training with lagged features
├── batch_inference_pipeline.py   # Generate predictions
├── dashboard.py                  # Streamlit dashboard
├── util.py                       # Helper functions
├── locations.json                # Skåne stations
└── .github/workflows/
    └── air-quality-daily.yml     # Automated daily runs (06:11 UTC)
```

---

## 🎯 Features
- Historical data backfill (1+ year)
- Daily automated feature collection
- XGBoost training pipeline
- Batch inference with predictions
- Dashboard with hindcast monitoring
- Added lagged PM2.5 (1, 2, 3 days ago)
- Lagged features capture temporal patterns, improving prediction accuracy
- 7 cities across Skåne region

---

## 📊 Model Performance

| Feature Set | R² Score | RMSE |
|-------------|----------|------|
| Weather only | -0.407 μg/m³ | 39.854 μg/m³|
| **+ Lagged** | -1.257 μg/m³| 21.028 μg/m³|

**Why lagged features help:** PM2.5 exhibits temporal autocorrelation - recent pollution levels predict future trends.

The high R² score for the lagged is explained by the pressence of a few extreme outliers of 999 μg/m³ for a few consecutive days. These can be improved by cleaning the data out of them.


---

## 🤖 Automated Pipeline

**Runs daily at 06:11 UTC via GitHub Actions:**
1. Fetch yesterday's air quality + weather
2. Get 7 day weather forecasts
3. Generate predictions
4. Update dashboard

---

## 📚 Data Sources & Stack

**Data:**
- Air Quality: [AQICN](https://aqicn.org/)
- Weather: [Open-Meteo](https://open-meteo.com/)

**Infrastructure:**
- Feature Store: [Hopsworks](https://www.hopsworks.ai/)
- Model: XGBoost with lagged features
- Orchestration: GitHub Actions
- Dashboard: Streamlit Cloud

---

## 🎓 Academic Context

**Course:** ID2223 Scalable Machine Learning and Deep Learning  
**Institution:** KTH Royal Institute of Technology  
**Based on:** [Serverless ML Course](https://github.com/featurestorebook/mlfs-book) by Prof. Jim Dowling

---