import os
from dotenv import load_dotenv
from hsfs.feature import Feature

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "AQIPredictor12")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

CITIES = [
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479},
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Hyderabad", "lat": 25.3960, "lon": 68.3578},
    {"name": "Lahore", "lat": 31.5204, "lon": 74.3587}
]

# Open-Meteo API Endpoints
WEATHER_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo Variables
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
    "cloud_cover"
]

AIR_QUALITY_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi"
]

# Explicit Schema to avoid inference mismatches
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
    Feature(name="target_us_aqi_72h", type="double")
]
