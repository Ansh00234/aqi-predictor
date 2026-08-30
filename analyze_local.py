import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from feature_pipeline.features import compute_features

def evaluate_regressor(y_true, y_pred, horizon: str):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

print("Loading local data...")
dfs = []
for city in ["Islamabad", "Karachi", "Hyderabad", "Lahore"]:
    df = pd.read_csv(f"local_backup/{city}_full.csv")
    df["time"] = pd.to_datetime(df["time"])
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)

# Split 80/20 randomly just like train_test_split
shuffled = full_df.sample(frac=1, random_state=42)
split_idx = int(len(shuffled) * 0.8)
train_df = shuffled.iloc[:split_idx]
test_df = shuffled.iloc[split_idx:]

y_train = train_df[["target_us_aqi_24h", "target_us_aqi_48h", "target_us_aqi_72h"]].fillna(0)
y_test = test_df[["target_us_aqi_24h", "target_us_aqi_48h", "target_us_aqi_72h"]].fillna(0)

cols_to_drop = ["city", "time", "target_us_aqi_24h", "target_us_aqi_48h", "target_us_aqi_72h"]
X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns]).fillna(0)
X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns]).fillna(0)

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
lstm_model.fit(X_train_lstm, y_train.values, epochs=10, batch_size=128, verbose=0)

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
