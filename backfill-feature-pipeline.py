# %%
import json
from pathlib import Path

import hopsworks
import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict
import util

# %%
class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    hopsworks_api_key: str
    aqicn_api_key: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

# %%
def load_locations(filepath: str = "locations.json") -> dict:
    """Load location data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


locations = load_locations()

# %%
def process_air_quality(df: pd.DataFrame, location: dict) -> None:
    """
    Process air quality dataframe in place.

    Transforms the dataframe to have columns: [id, date, pm25]
    """
    df.rename(columns={"median": "pm25"}, inplace=True)
    df["date"] = df["date"].dt.date
    df["pm25"] = df["pm25"].astype("float32")
    df.drop(df.columns.difference(["date", "pm25"]), axis=1, inplace=True)
    df.dropna(inplace=True)
    df["id"] = location["id"]

# %%
def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged PM2.5 features for 1, 2, and 3 days."""
    df = df.sort_values(['id', 'date']).copy()
    
    # Create lagged features for each location
    for location_id in df['id'].unique():
        mask = df['id'] == location_id
        location_data = df[mask].copy()
        
        # Create lagged features
        df.loc[mask, 'lagged_1'] = location_data['pm25'].shift(1)
        df.loc[mask, 'lagged_2'] = location_data['pm25'].shift(2)
        df.loc[mask, 'lagged_3'] = location_data['pm25'].shift(3)
    
    # Convert to float32
    for lag in [1, 2, 3]:
        df[f'lagged_{lag}'] = df[f'lagged_{lag}'].astype('float32')
    
    return df

# %%
def load_air_quality_data(locations: dict) -> pd.DataFrame:
    """Load and process air quality data for all locations."""
    dfs = []

    for location_id in locations:
        file_path = Path(f"data/{location_id}.csv")
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found")

        print(f"Processing {location_id}")
        df = pd.read_csv(
            file_path, comment="#", skipinitialspace=True, parse_dates=["date"]
        )
        process_air_quality(df, locations[location_id])
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Add lagged features
    print("Adding lagged features...")
    combined_df = add_lagged_features(combined_df)

    # Drop rows with NaN lagged features
    print(f"Rows before dropping NaNs: {len(combined_df)}")
    combined_df = combined_df.dropna(subset=['lagged_1', 'lagged_2', 'lagged_3'])
    print(f"Rows after dropping NaNs: {len(combined_df)}")
    
    return combined_df


air_quality_df = load_air_quality_data(locations)
air_quality_df.info()

# %%
project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
fs = project.get_feature_store()

# %%
air_quality_fg = fs.get_or_create_feature_group(
    name="air_quality",
    description="Air Quality characteristics of each day with lagged features",
    version=3,
    primary_key=["id"],
    event_time="date",
)
air_quality_fg

# %%
air_quality_fg.insert(air_quality_df)

# %%
air_quality_fg.update_feature_description("date", "Date of measurement of air quality")
air_quality_fg.update_feature_description("pm25", "Particles less than 2.5 micrometers in diameter (fine particles) pose health risk")
air_quality_fg.update_feature_description("lagged_1", "PM2.5 value from 1 day ago")
air_quality_fg.update_feature_description("lagged_2", "PM2.5 value from 2 days ago")
air_quality_fg.update_feature_description("lagged_3", "PM2.5 value from 3 days ago")

# %%
weather_df = util.get_historical(air_quality_df, locations)
weather_df.info()

# %%
weather_fg = fs.get_or_create_feature_group(
    name="weather",
    description="Weather characteristics of each day",
    version=2,
    primary_key=["id"],
    event_time="date",
)
weather_fg.insert(weather_df, wait=True)
# %%