import os, sys, joblib
import numpy as np
import pandas as pd
import hopsworks
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from feature_pipeline.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME,
    FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION
)

def evaluate_regressor(y_true, y_pred, horizon: str):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
fs = project.get_feature_store()
fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

print("Downloading data...")
X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)

cols_to_drop = ["city", "time"]
X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns]).fillna(0)
X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns]).fillna(0)
y_train = y_train.fillna(0)
y_test = y_test.fillna(0)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Ridge...")
ridge = MultiOutputRegressor(Ridge(alpha=1.0))
ridge.fit(X_train_scaled, y_train)

print("Training XGBoost...")
xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
xgb.fit(X_train, y_train)

print("Training LSTM...")
X_train_lstm = np.expand_dims(X_train_scaled, axis=1)
X_test_lstm = np.expand_dims(X_test_scaled, axis=1)

lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train_lstm.shape[2])),
    Dense(32, activation='relu'),
    Dense(3)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train.values, epochs=5, batch_size=128, verbose=1)

# Evaluate
ridge_preds = ridge.predict(X_test_scaled)
xgb_preds = xgb.predict(X_test)
lstm_preds = lstm_model.predict(X_test_lstm)

targets = ["24h", "48h", "72h"]
print("\n--- RESULTS ---")
for i, target in enumerate(targets):
    r_m = evaluate_regressor(y_test.iloc[:, i], ridge_preds[:, i], target)
    x_m = evaluate_regressor(y_test.iloc[:, i], xgb_preds[:, i], target)
    l_m = evaluate_regressor(y_test.iloc[:, i], lstm_preds[:, i], target)
    
    print(f"{target}:")
    print(f"  Ridge  : RMSE={r_m['rmse']:.2f}, MAE={r_m['mae']:.2f}, R2={r_m['r2']:.2f}")
    print(f"  XGBoost: RMSE={x_m['rmse']:.2f}, MAE={x_m['mae']:.2f}, R2={x_m['r2']:.2f}")
    print(f"  LSTM   : RMSE={l_m['rmse']:.2f}, MAE={l_m['mae']:.2f}, R2={l_m['r2']:.2f}")
