"""
Export all existing data from the Hopsworks aqi_features feature group
to local CSV files, one per city.

Output: local_backup/<CityName>_full.csv
"""

import os
import hopsworks
import pandas as pd
from feature_pipeline.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

BACKUP_DIR = "local_backup"


def export():
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    os.makedirs(BACKUP_DIR, exist_ok=True)

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Reading all data from feature group...")
    df = fg.read()
    print(f"Total rows: {len(df)}")

    if df.empty:
        print("No data found in feature group. Nothing to export.")
        return

    for city_name, city_df in df.groupby("city"):
        city_df = city_df.sort_values("time").reset_index(drop=True)
        path = os.path.join(BACKUP_DIR, f"{city_name}_full.csv")
        city_df.to_csv(path, index=False)
        print(f"  {city_name}: {len(city_df)} rows saved to {path}")

    print("Export complete.")


if __name__ == "__main__":
    export()
