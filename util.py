import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

OPENMETEO_DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "weather_code"
]


def _create_openmeteo_client(cache_expiry: int = -1) -> openmeteo_requests.Client:
    """Create an Open-Meteo API client with caching and retry logic."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=cache_expiry)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _process_weather_responses(
    responses: list, places: dict[str, dict], variables: list[str]
) -> pd.DataFrame:
    """Process weather API responses into a DataFrame."""
    dataframes = []

    for response, place_id in zip(responses, places.keys()):
        daily = response.Daily()
        data = {
            "id": place_id,
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ),
        }

        for idx, variable in enumerate(variables):
                data[variable] = daily.Variables(idx).ValuesAsNumpy()

        dataframes.append(pd.DataFrame(data))

    result = pd.concat(dataframes, ignore_index=True)
    result.dropna(inplace=True)
    result["date"] = result["date"].dt.date
    return result


def get_forecast(forecast_days: int, places: dict[str, dict]) -> pd.DataFrame:
    """Fetch weather forecast for multiple locations."""
    client = _create_openmeteo_client(cache_expiry=3600)

    params = {
        "latitude": [place["latitude"] for place in places.values()],
        "longitude": [place["longitude"] for place in places.values()],
        "forecast_days": forecast_days,
        "daily": OPENMETEO_DAILY_VARIABLES,
    }

    url = "https://api.open-meteo.com/v1/forecast"
    responses = client.weather_api(url, params=params)

    return _process_weather_responses(responses, places, OPENMETEO_DAILY_VARIABLES)


def get_historical(aq_df: pd.DataFrame, places: dict[str, dict]) -> pd.DataFrame:
    """Fetch historical weather data for date ranges present in air quality DataFrame."""
    start_dates = [
        aq_df[aq_df["id"] == place["id"]]["date"].min().strftime("%Y-%m-%d")
        for place in places.values()
    ]
    end_dates = [
        aq_df[aq_df["id"] == place["id"]]["date"].max().strftime("%Y-%m-%d")
        for place in places.values()
    ]

    return get_historical_in_daterange(start_dates, end_dates, places)


def get_historical_in_daterange(
    starts: list[str], ends: list[str], places: dict[str, dict]
) -> pd.DataFrame:
    """Fetch historical weather data for specified date ranges.

    Args:
        starts: Start dates for each location (in dict key order, format: YYYY-MM-DD)
        ends: End dates for each location (in dict key order, format: YYYY-MM-DD)
        places: Mapping of place IDs to location data (latitude, longitude)

    Returns:
        DataFrame with columns [id, date, ...weather_variables]
    """
    client = _create_openmeteo_client(cache_expiry=-1)

    params = {
        "latitude": [place["latitude"] for place in places.values()],
        "longitude": [place["longitude"] for place in places.values()],
        "start_date": starts,
        "end_date": ends,
        "daily": OPENMETEO_DAILY_VARIABLES,
    }

    url = "https://archive-api.open-meteo.com/v1/archive"
    responses = client.weather_api(url, params=params)

    return _process_weather_responses(responses, places, OPENMETEO_DAILY_VARIABLES)