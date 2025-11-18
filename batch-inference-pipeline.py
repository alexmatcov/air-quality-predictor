# %%
import json
import os
from datetime import date, timedelta
import hopsworks
import matplotlib.pyplot as plt
import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict
from xgboost import XGBRegressor

# %%
class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    hopsworks_api_key: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )

settings = Settings()

# %%
def load_locations(filepath: str = "locations.json") -> dict:
    """Load location data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


locations = load_locations()
locations

# %%
project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
fs = project.get_feature_store()
mr = project.get_model_registry()

# %%
# Retrieve the trained model
retrieved_model = mr.get_model(
    name="air_quality_xgboost_model",
    version=2,
)

print(f"✓ Retrieved model: {retrieved_model.name} v{retrieved_model.version}")

# %%
# Download model artifacts
saved_model_dir = retrieved_model.download()

# Load the XGBoost model
retrieved_xgboost_model = XGBRegressor()
retrieved_xgboost_model.load_model(os.path.join(saved_model_dir, "model.json"))
print("✓ Model loaded successfully")

# %%
# Get weather forecast data for future predictions
today_str = date.today().strftime("%Y-%m-%d")
weather_fg = fs.get_feature_group(name="weather", version=2)

batch_data = weather_fg.filter(weather_fg.date >= today_str).read()
print(f"✓ Retrieved {len(batch_data)} weather forecast records")
batch_data.head()

# %%
# Get historical air quality data for lagged features
lookback_days = 3 
lookback_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

# Try version 3 first (with lagged features from backfill)
try:
    air_quality_fg = fs.get_feature_group(name="air_quality", version=3)
    historical_aq = air_quality_fg.filter(air_quality_fg.date >= lookback_date).read()
    print(f"✓ Retrieved {len(historical_aq)} historical records from version 3")
except Exception as e:
    print(f"⚠ Could not load from version 3: {e}")
    print("Falling back to version 2...")
    air_quality_fg = fs.get_feature_group(name="air_quality", version=2)
    historical_aq = air_quality_fg.filter(air_quality_fg.date >= lookback_date).read()
    print(f"✓ Retrieved {len(historical_aq)} historical records from version 2")

# Sort by date for easy lookup
historical_aq = historical_aq.sort_values(['id', 'date'])
print(f"Date range: {historical_aq['date'].min()} to {historical_aq['date'].max()}")
historical_aq.head()

# %%
# Prepare predictions with lagged features from historical data (simplified approach)
predictions_list = []

for location_id, location in locations.items():
    location_weather = batch_data[batch_data["id"] == location_id].copy()
    
    if location_weather.empty:
        print(f"⚠ No weather forecast for {location['city']}")
        continue
    
    # Get the most recent 3 days of historical data for this location
    location_history = historical_aq[historical_aq['id'] == location_id].sort_values('date', ascending=False)
    
    if len(location_history) < 3:
        print(f"⚠ Not enough historical data for {location['city']} - need at least 3 days")
        continue
    
    # Use the 3 most recent historical values as lagged features for ALL predictions
    lagged_1 = float(location_history.iloc[0]['pm25'])  # Most recent
    lagged_2 = float(location_history.iloc[1]['pm25'])  # 2 days ago
    lagged_3 = float(location_history.iloc[2]['pm25'])  # 3 days ago
    
    print(f"\nProcessing forecasts for {location['city']}...")
    print(f"  Using lagged values: {lagged_1:.1f}, {lagged_2:.1f}, {lagged_3:.1f}")
    
    # For each forecast day, use the same lagged features
    for _, weather_row in location_weather.iterrows():
        forecast_date = pd.to_datetime(weather_row["date"])
        
        # Prepare feature vector in the SAME ORDER as training
        feature_row = {
            "lagged_1": lagged_1,
            "lagged_2": lagged_2,
            "lagged_3": lagged_3,
            "weather_temperature_2m_mean": weather_row["temperature_2m_mean"],
            "weather_precipitation_sum": weather_row["precipitation_sum"],
            "weather_wind_speed_10m_max": weather_row["wind_speed_10m_max"],
            "weather_wind_direction_10m_dominant": weather_row["wind_direction_10m_dominant"]
        }
        
        X_pred = pd.DataFrame([feature_row])
        
        # Make prediction
        prediction = retrieved_xgboost_model.predict(X_pred)[0]
        prediction = max(0, prediction)  # Clip negative values
        
        predictions_list.append({
            "id": location_id,
            "date": weather_row["date"],
            "predicted_pm25": prediction,
            "forecast_date": date.today()
        })
        
        print(f"  ✓ {forecast_date.date()}: {prediction:.2f} μg/m³")

# %%
forecast_data = pd.DataFrame(predictions_list)

# Convert predicted_pm25 to float32 to match feature group schema
forecast_data["predicted_pm25"] = forecast_data["predicted_pm25"].astype("float32")

print(f"\n✓ Generated {len(forecast_data)} total predictions")
forecast_data.head()

# %%
# Save predictions to feature group
if not forecast_data.empty:
    forecasts_fg = fs.get_or_create_feature_group(
        name="air_quality_forecasts",
        description="Daily air quality predictions with lagged features",
        version=1,
        primary_key=["id", "forecast_date"],
        event_time="date",
    )

    forecasts_fg.insert(forecast_data, write_options={"wait_for_job": True})
    print("✓ Predictions saved to Hopsworks")
else:
    print("⚠ No predictions to save - check if historical data is available")

# %%
# Create forecast plots for each location
images_dir = "air_quality_model/images/forecasts"
os.makedirs(images_dir, exist_ok=True)

for location_id, location in locations.items():
    location_forecast = forecast_data[forecast_data["id"] == location_id].copy()
    
    if location_forecast.empty:
        print(f"⚠ No forecast data for {location['city']}")
        continue
    
    location_forecast = location_forecast.sort_values("date")
    
    plt.figure(figsize=(12, 6))
    plt.plot(location_forecast["date"], location_forecast["predicted_pm25"], 
             marker="o", linewidth=2, markersize=8, label="Predicted PM2.5 (with lagged features)")
    plt.xlabel("Date")
    plt.ylabel("Predicted PM2.5 (μg/m³)")
    plt.title(f"Air Quality Forecast - {location['city']}, {location['country']}")
    plt.axhline(y=25, color='orange', linestyle='--', label='Moderate', alpha=0.7)
    plt.axhline(y=50, color='red', linestyle='--', label='Unhealthy', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(images_dir, f"pm25_forecast_{location['city']}.png")
    plt.savefig(plot_path, dpi=100)
    plt.close()
    
    print(f"✓ Saved forecast plot for {location['city']}")

# %%
print(f"""
Batch Inference Summary (with Lagged Features):
----------------------------------------------
Forecast date: {date.today()}
Locations: {len(locations)}
Total predictions: {len(predictions_list)}
Predictions per location: ~{len(predictions_list) // len(locations) if locations else 0}
Saved to feature group: air_quality_forecasts v1

Note: Predictions use the 3 most recent historical PM2.5 values as lagged features
      for all forecast days. These values automatically update as the daily pipeline runs.
""")

# %%