import pandas as pd
import hopsworks
import datetime
import logging
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from feature_pipeline.config import (
    CITIES, 
    HOPSWORKS_API_KEY, 
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_GROUP_SCHEMA
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

def backfill():
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")
        
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    
    logger.info("Checking for already backfilled cities in Hopsworks...")
    existing_cities = []
    try:
        aqi_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        existing_data = aqi_fg.select(['city']).read()
        existing_cities = existing_data['city'].unique().tolist()
        logger.info(f"Found existing cities in Hopsworks: {existing_cities}")
    except Exception as e:
        logger.info(f"Feature group not found or couldn't be read: {e}. Creating fresh.")
        aqi_fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description="Air Quality Index features",
            primary_key=["city", "time"],
            event_time="time",
            features=FEATURE_GROUP_SCHEMA
        )
        logger.info("Waiting 15s for Hopsworks to fully initialize the feature group...")
        time.sleep(15)
        
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=730)  # 2 years
    
    for city in CITIES:
        if city['name'] in existing_cities:
            logger.info(f"Skipping {city['name']}, already backfilled.")
            continue
            
        logger.info(f"Processing backfill for {city['name']}...")
        
        # Generate monthly date ranges for chunked fetching + insertion
        chunk_start = start_date
        chunk_idx = 0
        while chunk_start < end_date:
            chunk_end = min(chunk_start + datetime.timedelta(days=30), end_date)
            chunk_idx += 1
            
            logger.info(f"  [{city['name']}] Chunk {chunk_idx}: {chunk_start} to {chunk_end}")
            
            try:
                df_raw = fetch_all_data(
                    city, 
                    chunk_start.strftime("%Y-%m-%d"), 
                    chunk_end.strftime("%Y-%m-%d"), 
                    is_historical=True
                )
                df_features = compute_features(df_raw, is_training=True)
                
                if df_features['time'].dt.tz is not None:
                    df_features['time'] = df_features['time'].dt.tz_localize(None)
                
                logger.info(f"  [{city['name']}] Inserting {len(df_features)} rows...")
                insert_with_retry(aqi_fg, df_features)
                logger.info(f"  [{city['name']}] Chunk {chunk_idx} inserted successfully.")
            except Exception as e:
                logger.error(f"  [{city['name']}] Chunk {chunk_idx} failed: {e}")
                raise
            
            chunk_start = chunk_end + datetime.timedelta(days=1)

        logger.info(f"Successfully backfilled {city['name']}.")

    logger.info("Backfill completed successfully.")

if __name__ == "__main__":
    backfill()
