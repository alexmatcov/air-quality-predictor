# %%
import json
from pathlib import Path

import hopsworks
import pandas as pd
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

# %%
def process_air_quality(df: pd.DataFrame, place: dict) -> None:
    """
    Process air quality dataframe in place.

    Transforms the dataframe to have columns: [id, date, pm25]
    """
    df.rename(columns={"median": "pm25"}, inplace=True)
    df["date"] = df["date"].dt.date
    df["pm25"] = df["pm25"].astype("float32")
    df.drop(df.columns.difference(["date", "pm25"]), axis=1, inplace=True)
    df.dropna(inplace=True)
    df["id"] = place["id"]


def load_air_quality_data(locations: dict) -> pd.DataFrame:
    """Load and process air quality data for all locations."""
    dfs = []

    for place_id in locations:
        file_path = Path(f"data/air-quality/{place_id}.csv")
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found")

        print(f"Processing {place_id}")
        df = pd.read_csv(
            file_path, comment="#", skipinitialspace=True, parse_dates=["date"]
        )
        process_air_quality(df, locations[place_id])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


air_quality_df = load_air_quality_data(locations)
air_quality_df.head()

# %%
project = hopsworks.login()
fs = project.get_feature_store()

# %%
air_quality_fg = fs.get_or_create_feature_group(
    name="air_quality",
    description="Air Quality characteristics of each day",
    version=2,
    primary_key=["id"],
    event_time="date",
)
air_quality_fg.insert(air_quality_df)
air_quality_fg.update_feature_description("date", "Date of measurement of air quality")
air_quality_fg.update_feature_description(
    "pm25",
    "Particles less than 2.5 micrometers in diameter (fine particles) pose health risk",
)

# %%
weather_df = util.get_historical(air_quality_df, locations)
weather_df.head()

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
