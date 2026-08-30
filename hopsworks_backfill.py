"""
AQI Predictor: Historical Backfill Script
Run this script inside Hopsworks Jupyter (Sidebar > Jupyter).
It is fully self-contained with no local imports needed.

It fetches 2 years of hourly weather + air quality data from Open-Meteo
for 4 Pakistani cities, engineers features, and inserts into the
aqi_features feature group in monthly chunks.

Resume-safe: skips fully backfilled cities and already-inserted chunks.
"""

import os
import requests
import pandas as pd
import numpy as np
import datetime
import time
import hopsworks
from hsfs.feature import Feature

# Config

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

CITIES = [
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479},
    {"name": "Karachi",   "lat": 24.8607, "lon": 67.0011},
    {"name": "Hyderabad", "lat": 25.3960, "lon": 68.3578},
    {"name": "Lahore",    "lat": 31.5204, "lon": 74.3587},
]

WEATHER_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "surface_pressure", "cloud_cover",
]

AIR_QUALITY_VARIABLES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi",
]

FEATURE_GROUP_SCHEMA = [
    Feature(name="city", type="string"),
    Feature(name="time", type="timestamp"),
    Feature(name="temperature_2m", type="double"),
    Feature(name="relative_humidity_2m", type="bigint"),
    Feature(name="wind_speed_10m", type="double"),
    Feature(name="wind_direction_10m", type="bigint"),
    Feature(name="precipitation", type="double"),
    Feature(name="surface_pressure", type="double"),
    Feature(name="cloud_cover", type="bigint"),
    Feature(name="pm10", type="double"),
    Feature(name="pm2_5", type="double"),
    Feature(name="carbon_monoxide", type="double"),
    Feature(name="nitrogen_dioxide", type="double"),
    Feature(name="sulphur_dioxide", type="double"),
    Feature(name="ozone", type="double"),
    Feature(name="us_aqi", type="bigint"),
    Feature(name="hour", type="bigint"),
    Feature(name="day_of_week", type="bigint"),
    Feature(name="month", type="bigint"),
    Feature(name="is_weekend", type="bigint"),
    Feature(name="season", type="bigint"),
    Feature(name="us_aqi_lag_1", type="double"),
    Feature(name="pm2_5_lag_1", type="double"),
    Feature(name="us_aqi_lag_3", type="double"),
    Feature(name="pm2_5_lag_3", type="double"),
    Feature(name="us_aqi_lag_6", type="double"),
    Feature(name="pm2_5_lag_6", type="double"),
    Feature(name="us_aqi_lag_24", type="double"),
    Feature(name="pm2_5_lag_24", type="double"),
    Feature(name="us_aqi_lag_48", type="double"),
    Feature(name="pm2_5_lag_48", type="double"),
    Feature(name="us_aqi_diff_1", type="double"),
    Feature(name="us_aqi_roll_min_6", type="double"),
    Feature(name="us_aqi_roll_max_6", type="double"),
    Feature(name="us_aqi_roll_mean_6", type="double"),
    Feature(name="us_aqi_roll_std_6", type="double"),
    Feature(name="pm2_5_roll_mean_6", type="double"),
    Feature(name="us_aqi_roll_min_24", type="double"),
    Feature(name="us_aqi_roll_max_24", type="double"),
    Feature(name="us_aqi_roll_mean_24", type="double"),
    Feature(name="us_aqi_roll_std_24", type="double"),
    Feature(name="pm2_5_roll_mean_24", type="double"),
    Feature(name="us_aqi_roll_min_168", type="double"),
    Feature(name="us_aqi_roll_max_168", type="double"),
    Feature(name="us_aqi_roll_mean_168", type="double"),
    Feature(name="us_aqi_roll_std_168", type="double"),
    Feature(name="pm2_5_roll_mean_168", type="double"),
    Feature(name="target_us_aqi_24h", type="double"),
    Feature(name="target_us_aqi_48h", type="double"),
    Feature(name="target_us_aqi_72h", type="double"),
]

LOCAL_BACKUP_DIR = "local_backup"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds

# Data Fetching (with retry logic)

def _request_with_retry(url, params, label="API"):
    """Makes a GET request with retry logic and exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            if attempt == MAX_RETRIES:
                print(f"    {label} request failed after {MAX_RETRIES} attempts: {ex}")
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"    {label} request failed (attempt {attempt}/{MAX_RETRIES}): {ex}. Retrying in {delay}s...")
            time.sleep(delay)

def fetch_weather(city, start_date, end_date):
    params = {
        "latitude": city["lat"], "longitude": city["lon"],
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES), "timezone": "auto",
    }
    data = _request_with_retry(WEATHER_HISTORICAL_URL, params, label="Weather")
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

def fetch_air_quality(city, start_date, end_date):
    params = {
        "latitude": city["lat"], "longitude": city["lon"],
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_VARIABLES), "timezone": "auto",
    }
    data = _request_with_retry(AIR_QUALITY_URL, params, label="AirQuality")
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

def fetch_all_data(city, start_date, end_date):
    weather = fetch_weather(city, start_date, end_date)
    aqi = fetch_air_quality(city, start_date, end_date)
    df = pd.merge(weather, aqi, on="time", how="inner")
    df["city"] = city["name"]
    return df.sort_values("time").reset_index(drop=True)

# Feature Engineering

def compute_features(df):
    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    month_to_season = {12:1,1:1,2:1, 3:2,4:2,5:2, 6:3,7:3,8:3,9:3, 10:4,11:4}
    df["season"] = df["month"].map(month_to_season)

    for c in ["hour", "day_of_week", "month", "is_weekend", "season"]:
        df[c] = df[c].astype("int64")

    df = df.sort_values("time").reset_index(drop=True)
    grouped = df.groupby("city")

    for lag in [1, 3, 6, 24, 48]:
        df[f"us_aqi_lag_{lag}"] = grouped["us_aqi"].shift(lag)
        df[f"pm2_5_lag_{lag}"] = grouped["pm2_5"].shift(lag)

    df["us_aqi_diff_1"] = df["us_aqi"] - df["us_aqi_lag_1"]

    for w in [6, 24, 168]:
        df[f"us_aqi_roll_min_{w}"]  = grouped["us_aqi"].rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
        df[f"us_aqi_roll_max_{w}"]  = grouped["us_aqi"].rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
        df[f"us_aqi_roll_mean_{w}"] = grouped["us_aqi"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"us_aqi_roll_std_{w}"]  = grouped["us_aqi"].rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
        df[f"pm2_5_roll_mean_{w}"]  = grouped["pm2_5"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

    df["target_us_aqi_24h"] = grouped["us_aqi"].shift(-24)
    df["target_us_aqi_48h"] = grouped["us_aqi"].shift(-48)
    df["target_us_aqi_72h"] = grouped["us_aqi"].shift(-72)
    df = df.dropna(subset=["target_us_aqi_72h"])

    df = df.fillna(method="ffill").fillna(0)
    return df

# Resume Helpers

def get_existing_city_time_ranges(fg):
    """Query existing data to determine which cities and date ranges are already inserted."""
    try:
        existing = fg.select(["city", "time"]).read()
        if existing.empty:
            return {}
        result = {}
        for city_name, grp in existing.groupby("city"):
            result[city_name] = {
                "min_time": grp["time"].min(),
                "max_time": grp["time"].max(),
                "count": len(grp),
            }
        return result
    except Exception as ex:
        print(f"Could not read existing data (feature group may be empty): {ex}")
        return {}

def chunk_already_covered(city_name, chunk_start_str, chunk_end_str, existing_ranges):
    """Check if a chunk's date range is already fully covered by existing data."""
    if city_name not in existing_ranges:
        return False
    info = existing_ranges[city_name]
    chunk_start = pd.Timestamp(chunk_start_str)
    chunk_end = pd.Timestamp(chunk_end_str)
    return info["min_time"] <= chunk_start and info["max_time"] >= chunk_end

# Backfill

project = hopsworks.login()  # No API key needed inside Hopsworks Jupyter
fs = project.get_feature_store()

# The primary_key=["city", "time"] ensures Hopsworks treats (city, time) as a
# composite primary key. Duplicate rows with the same city+time will be upserted,
# not duplicated.
print("Getting or creating feature group with explicit schema...")
aqi_fg = fs.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="Air Quality Index features for Pakistan cities",
    primary_key=["city", "time"],
    event_time="time",
    features=FEATURE_GROUP_SCHEMA,
)
time.sleep(15)

# Check what already exists for resume
print("Checking existing data for resume...")
existing_ranges = get_existing_city_time_ranges(aqi_fg)
if existing_ranges:
    for city_name, info in existing_ranges.items():
        print(f"  {city_name}: {info['count']} rows, {info['min_time']} to {info['max_time']}")
else:
    print("  No existing data found. Starting fresh.")

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=730)

for city in CITIES:
    city_name = city["name"]

    # Skip fully backfilled cities (those covering the entire date range)
    if city_name in existing_ranges:
        info = existing_ranges[city_name]
        expected_start = pd.Timestamp(start_date)
        expected_end = pd.Timestamp(end_date)
        if info["min_time"] <= expected_start + pd.Timedelta(days=1) and info["max_time"] >= expected_end - pd.Timedelta(days=2):
            print(f"\n{city_name}: Already fully backfilled ({info['count']} rows). Skipping.")
            continue

    print(f"\nBackfilling: {city_name}")

    chunk_start = start_date
    chunk_idx = 0

    while chunk_start < end_date:
        chunk_end = min(chunk_start + datetime.timedelta(days=30), end_date)
        chunk_idx += 1

        s = chunk_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        # Skip chunks already covered
        if chunk_already_covered(city_name, s, e, existing_ranges):
            print(f"  [{city_name}] Chunk {chunk_idx}: {s} to {e} ... SKIPPED (already exists)")
            chunk_start = chunk_end + datetime.timedelta(days=1)
            continue

        print(f"  [{city_name}] Chunk {chunk_idx}: {s} to {e} ... ", end="")

        try:
            raw = fetch_all_data(city, s, e)
            feat = compute_features(raw)

            if feat["time"].dt.tz is not None:
                feat["time"] = feat["time"].dt.tz_localize(None)

            aqi_fg.insert(feat, write_options={"wait_for_job": True})

            # Save chunk to local CSV backup
            os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)
            backup_path = os.path.join(LOCAL_BACKUP_DIR, f"{city_name}_chunk_{chunk_idx}_{s}_to_{e}.csv")
            feat.to_csv(backup_path, index=False)

            print(f"OK ({len(feat)} rows, saved to {backup_path})")
        except Exception as ex:
            print(f"FAILED: {ex}")
            raise

        chunk_start = chunk_end + datetime.timedelta(days=1)

    print(f"  {city_name} done.")

print("\nHistorical backfill complete!")
