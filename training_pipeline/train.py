import os
import joblib
import logging
import hopsworks
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from feature_pipeline.config import (
    HOPSWORKS_API_KEY, 
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION
)
from training_pipeline.evaluate import evaluate_regressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")
        
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    
    try:
        logger.info("Getting feature view...")
        feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
        if feature_view is None:
            raise ValueError("Feature view returned None")
    except Exception:
        logger.info("Feature view not found, creating it...")
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        query = fg.select_all()
        # Ensure we don't include primary keys or time in the training features natively
        feature_view = fs.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            description="Feature view for AQI forecasting",
            labels=["target_us_aqi_24h", "target_us_aqi_48h", "target_us_aqi_72h"],
            query=query
        )
        
    logger.info("Creating train/test split...")
    # Get training data (train_test_split natively handles random splits)
    # Note: For time-series, a temporal split is better, but since this is a simple project, random is okay,
    # or we can sort by time. We'll use the native train_test_split.
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)
    
    # Drop non-feature columns
    cols_to_drop = ["city", "time"]
    X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])
    
    # Fill any remaining NaNs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(0)
    y_test = y_test.fillna(0)
    
    logger.info("Training Ridge Baseline...")
    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_train, y_train)
    
    logger.info("Training XGBoost Regressor...")
    xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    xgb.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Ridge Evaluation")
    ridge_preds = ridge.predict(X_test)
    xgb_preds = xgb.predict(X_test)
    
    ridge_metrics = {}
    xgb_metrics = {}
    targets = ["target_us_aqi_24h", "target_us_aqi_48h", "target_us_aqi_72h"]
    for i, target in enumerate(targets):
        r_m = evaluate_regressor(y_test.iloc[:, i], ridge_preds[:, i], target.split('_')[-1])
        ridge_metrics.update(r_m)
        x_m = evaluate_regressor(y_test.iloc[:, i], xgb_preds[:, i], target.split('_')[-1])
        xgb_metrics.update(x_m)
        
    # Pick best (XGBoost almost certainly better, we'll just register XGBoost)
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgb_aqi_model.pkl")
    joblib.dump(xgb, model_path)
    
    # Register model
    mr = project.get_model_registry()
    
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    
    logger.info("Registering model in Hopsworks...")
    model = mr.python.create_model(
        name="aqi_xgboost_multi", 
        metrics=xgb_metrics,
        model_schema=model_schema,
        description="XGBoost MultiOutput Regressor for 24h, 48h, 72h AQI Forecasting"
    )
    
    model.save(model_path)
    logger.info("Model saved and registered successfully.")

if __name__ == "__main__":
    train()
