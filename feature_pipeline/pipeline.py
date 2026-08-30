import pandas as pd
import hopsworks
import datetime
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from feature_pipeline.config import (
    CITIES, 
    HOPSWORKS_API_KEY, 
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION
)
from feature_pipeline.fetch_data import fetch_all_data
from feature_pipeline.features import compute_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    wait=wait_exponential(multiplier=10, min=10, max=60), 
    stop=stop_after_attempt(4), 
    before_sleep=lambda retry_state: logger.warning(f"Retrying Hopsworks insert... Attempt {retry_state.attempt_number}")
)
def insert_with_retry(fg, df):
    fg.insert(df, write_options={"wait_for_job": True})

def run_hourly():
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")
        
    # We fetch a bit of history (e.g. 7 days) to compute rolling and lag features properly for the current hour
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    all_cities_data = []
    
    for city in CITIES:
        logger.info(f"Processing hourly update for {city['name']}...")
        df_raw = fetch_all_data(city, start_date, end_date, is_historical=False)
        # For hourly, we don't compute forward looking targets since they haven't happened yet!
        # Wait, for the feature store, we usually just write the features.
        # But if the schema expects the target columns, we need to append them as NaN.
        # Let's compute them just to keep the schema aligned, they will be NaN for the latest records.
        df_features = compute_features(df_raw, is_training=False)
        
        # Add empty target columns to match schema
        df_features["target_us_aqi_24h"] = float('nan')
        df_features["target_us_aqi_48h"] = float('nan')
        df_features["target_us_aqi_72h"] = float('nan')
        
        # Only keep the last 24 hours to insert to Hopsworks (to avoid re-inserting 7 days every hour)
        # Actually, Hopsworks handles upserts based on primary key (city, time). So inserting 7 days is fine and safer.
        all_cities_data.append(df_features)
        
    final_df = pd.concat(all_cities_data, ignore_index=True)
    
    if final_df['time'].dt.tz is not None:
        final_df['time'] = final_df['time'].dt.tz_localize(None)
        
    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    
    aqi_fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )
    
    logger.info("Inserting hourly data into feature group...")
    insert_with_retry(aqi_fg, final_df)
    
    logger.info("Hourly pipeline completed successfully.")

if __name__ == "__main__":
    run_hourly()
