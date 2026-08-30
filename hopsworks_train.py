"""
AQI Predictor: Model Training Script for Hopsworks Jupyter
Run this script inside Hopsworks Jupyter (Sidebar > Jupyter).
It is fully self-contained with no local imports needed.

It creates a feature view from the aqi_features group,
creates a train/test split, trains Ridge and XGBoost baseline models,
evaluates them, and registers the best model (XGBoost) into
the Hopsworks Model Registry.
"""

import os
import joblib
import logging
import hopsworks
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

def evaluate_regressor(y_true, y_pred, horizon: str):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        f"rmse_{horizon}": rmse,
        f"mae_{horizon}": mae,
        f"r2_{horizon}": r2
    }
    
    logger.info(f"Evaluation for {horizon}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
    return metrics

def train():
    project = hopsworks.login() # No API key needed in Hopsworks Jupyter
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
    
    logger.info("Scaling features for Deep Learning and Ridge...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train_lstm = np.expand_dims(X_train_scaled, axis=1)
    X_test_lstm = np.expand_dims(X_test_scaled, axis=1)
    
    logger.info("Training LSTM (Deep Learning)...")
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(1, X_train_lstm.shape[2])),
        Dense(32, activation='relu'),
        Dense(3) # 3 outputs for 24h, 48h, 72h
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train.values, epochs=10, batch_size=128, validation_split=0.1, verbose=0)
    
    logger.info("Training Ridge Baseline...")
    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_train_scaled, y_train)
    
    logger.info("Training XGBoost Regressor...")
    xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    xgb.fit(X_train, y_train) # XGBoost typically doesn't need scaling
    
    # Evaluate
    logger.info("Model Evaluation")
    ridge_preds = ridge.predict(X_test_scaled)
    xgb_preds = xgb.predict(X_test)
    lstm_preds = lstm_model.predict(X_test_lstm)
    
    ridge_metrics = {}
    xgb_metrics = {}
    lstm_metrics = {}
    targets = ["target_us_aqi_24h", "target_us_aqi_48h", "target_us_aqi_72h"]
    for i, target in enumerate(targets):
        logger.info(f"--- {target} ---")
        logger.info("Ridge:")
        r_m = evaluate_regressor(y_test.iloc[:, i], ridge_preds[:, i], target.split('_')[-1])
        ridge_metrics.update(r_m)
        logger.info("XGBoost:")
        x_m = evaluate_regressor(y_test.iloc[:, i], xgb_preds[:, i], target.split('_')[-1])
        xgb_metrics.update(x_m)
        logger.info("LSTM:")
        l_m = evaluate_regressor(y_test.iloc[:, i], lstm_preds[:, i], target.split('_')[-1])
        lstm_metrics.update(l_m)
        
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
