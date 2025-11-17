# %%
import json
import hopsworks
import pandas as pd
from datetime import date
from pydantic_settings import BaseSettings
import util

# %%
class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    hopsworks_api_key: str
    aqicn_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()

# %%
def load_locations(filepath: str = "locations.json") -> dict:
    """Load location data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


locations = load_locations()
locations

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
weather_df = util.get_forecast(forecast_days=7, places=locations)
weather_df.info()

# %%
project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
fs = project.get_feature_store()

# %%
air_quality_fg = fs.get_feature_group(name="air_quality", version=2)
weather_fg = fs.get_feature_group(name="weather", version=2)

# %%
print(f"Inserting {len(air_quality_df)} air quality records...")
air_quality_fg.insert(air_quality_df)

# %%
print(f"Inserting {len(weather_df)} weather records...")
weather_fg.insert(weather_df, wait=True)
print("✓ Daily feature pipeline completed successfully!")

# %%