import hopsworks
from feature_pipeline.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from feature_pipeline.fetch_data import fetch_all_data
from feature_pipeline.features import compute_features

def main():
    # 1. Print dtypes
    print("DTYPES OF BACKFILL DATAFRAME")
    city = {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479}
    df_raw = fetch_all_data(city, "2024-08-01", "2024-08-05", is_historical=True)
    df_features = compute_features(df_raw, is_training=True)
    for col, dt in df_features.dtypes.items():
        print(f"{col}: {dt}")
    
    # 2. Delete Feature Group
    print(f"\nDELETING FEATURE GROUP {FEATURE_GROUP_NAME}_{FEATURE_GROUP_VERSION}")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    try:
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        fg.delete()
        print("Feature group deleted successfully.")
    except Exception as e:
        print(f"Error deleting feature group (it might not exist): {e}")

if __name__ == "__main__":
    main()
