import requests
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from feature_pipeline.config import (
    WEATHER_HISTORICAL_URL,
    WEATHER_FORECAST_URL,
    AIR_QUALITY_URL,
    WEATHER_VARIABLES,
    AIR_QUALITY_VARIABLES
)
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=30), 
    stop=stop_after_attempt(5), 
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=lambda retry_state: logger.warning(f"Retrying Open-Meteo API request... Attempt {retry_state.attempt_number}")
)
def fetch_weather_data(city: dict, start_date: str, end_date: str, is_historical: bool = True) -> pd.DataFrame:
    """Fetches hourly weather data for a given city."""
    url = WEATHER_HISTORICAL_URL if is_historical else WEATHER_FORECAST_URL
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=30), 
    stop=stop_after_attempt(5), 
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=lambda retry_state: logger.warning(f"Retrying Open-Meteo AQI request... Attempt {retry_state.attempt_number}")
)
def fetch_air_quality_data(city: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches hourly air quality data for a given city."""
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_VARIABLES),
        "timezone": "auto"
    }
    
    response = requests.get(AIR_QUALITY_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

def fetch_all_data(city: dict, start_date: str, end_date: str, is_historical: bool = True) -> pd.DataFrame:
    """Fetches and merges weather and air quality data."""
    logger.info(f"Fetching data for {city['name']} from {start_date} to {end_date}...")
    
    weather_df = fetch_weather_data(city, start_date, end_date, is_historical)
    aqi_df = fetch_air_quality_data(city, start_date, end_date)
    
    # Merge on time
    df = pd.merge(weather_df, aqi_df, on="time", how="inner")
    df["city"] = city["name"]
    
    # Sort just in case
    df = df.sort_values("time").reset_index(drop=True)
    return df
