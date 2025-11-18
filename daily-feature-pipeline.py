# %%
import json
from datetime import date, timedelta
import hopsworks
import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict
import util

# %%
class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    hopsworks_api_key: str
    aqicn_api_key: str
    
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
def add_lagged_features(air_quality_df: pd.DataFrame, air_quality_fg) -> pd.DataFrame:
    """Add lagged PM2.5 features for the previous 1, 2, and 3 days."""
    today = date.today()
    
    # Get last 4 days of data to calculate lags
    last_4_days = (today - timedelta(days=4)).strftime("%Y-%m-%d")
    historical_data = air_quality_fg.filter(air_quality_fg.date >= last_4_days).read()
    
    # For each row in today's data, find lagged values
    for idx, row in air_quality_df.iterrows():
        location_id = row['id']
        current_date = row['date']
        
        # Get historical data for this location
        location_history = historical_data[historical_data['id'] == location_id].sort_values('date')
        
        # Calculate lagged features
        for lag in [1, 2, 3]:
            lag_date = pd.to_datetime(current_date) - timedelta(days=lag)
            lag_data = location_history[pd.to_datetime(location_history['date']) == lag_date]
            
            if not lag_data.empty:
                air_quality_df.loc[idx, f'lagged_{lag}'] = lag_data.iloc[0]['pm25']
            else:
                air_quality_df.loc[idx, f'lagged_{lag}'] = None
    
    # Convert lagged columns to float32
    for lag in [1, 2, 3]:
        if f'lagged_{lag}' in air_quality_df.columns:
            air_quality_df[f'lagged_{lag}'] = air_quality_df[f'lagged_{lag}'].astype('float32')
    
    return air_quality_df


# %%
air_quality_df = pd.DataFrame()
today = date.today()

for location_id, location in locations.items():
    try:
        aq_data = util.get_pm25(location_id, location, today, settings.aqicn_api_key)
        air_quality_df = pd.concat([air_quality_df, aq_data], ignore_index=True)
        print(f"✓ Fetched air quality for {location['city']}")
    except Exception as e:
        print(f"✗ Error for {location_id}: {e}")
        continue

air_quality_df.info()

# %%
weather_df = util.get_forecast(forecast_days=10, places=locations)
weather_df.info()

# %%
project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
fs = project.get_feature_store()

# %%
air_quality_fg = fs.get_feature_group(name="air_quality", version=3)
weather_fg = fs.get_feature_group(name="weather", version=2)

# %%
# Add lagged features before inserting
print("Adding lagged features...")
air_quality_df = add_lagged_features(air_quality_df, air_quality_fg)
print("✓ Lagged features added")
air_quality_df.info()

# %%
print(f"Inserting {len(air_quality_df)} air quality records...")
air_quality_fg.insert(air_quality_df)

# %%
print(f"Inserting {len(weather_df)} weather records...")
weather_fg.insert(weather_df, wait=True)
print("✓ Daily feature pipeline completed successfully!")

# %%