import streamlit as st
import pandas as pd
import hopsworks
import datetime
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add parent dir to path so we can import feature_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_pipeline.config import (
    CITIES, 
    HOPSWORKS_API_KEY, 
    HOPSWORKS_PROJECT_NAME
)
from feature_pipeline.fetch_data import fetch_all_data
from feature_pipeline.features import compute_features
from training_pipeline.evaluate import get_aqi_category

# Set page config
st.set_page_config(page_title="AQI Predictor", layout="wide")

st.title("Pakistan AQI 3-Day Forecast")
st.markdown("Predicting Air Quality Index for major cities using XGBoost & Open-Meteo Data.")

@st.cache_resource
def get_model():
    if not HOPSWORKS_API_KEY:
        st.error("HOPSWORKS_API_KEY not found in environment.")
        st.stop()
        
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    mr = project.get_model_registry()
    try:
        model = mr.get_model("aqi_xgboost_multi", version=1)
        model_dir = model.download()
        return joblib.load(os.path.join(model_dir, "xgb_aqi_model.pkl"))
    except Exception as e:
        st.warning(f"Could not load model from Hopsworks: {e}. Ensure the training pipeline has run.")
        return None

# Sidebar
st.sidebar.header("Settings")
city_names = [c["name"] for c in CITIES]
selected_city_name = st.sidebar.selectbox("Select City", city_names)
selected_city = next(c for c in CITIES if c["name"] == selected_city_name)

# Main flow
model = get_model()

if model:
    with st.spinner(f"Fetching real-time data for {selected_city_name}..."):
        # Fetch last 7 days of data for features
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        
        raw_df = fetch_all_data(selected_city, start_date, end_date, is_historical=False)
        features_df = compute_features(raw_df, is_training=False)
        
        # Current status
        current_data = features_df.iloc[-1]
        current_aqi = current_data["us_aqi"]
        category = get_aqi_category(current_aqi)
        
        # Hazard Banner
        colors = {
            "Good": "#00e400", "Moderate": "#ffff00", 
            "Unhealthy for Sensitive Groups": "#ff7e00",
            "Unhealthy": "#ff0000", "Very Unhealthy": "#8f3f97", "Hazardous": "#7e0023"
        }
        text_color = "black" if category in ["Good", "Moderate"] else "white"
        st.markdown(f"""
        <div style="background-color: {colors[category]}; padding: 15px; border-radius: 5px; text-align: center; color: {text_color};">
            <h2 style="margin:0; color:{text_color};">Current AQI: {current_aqi:.1f} - {category}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.write("### Current Pollutants")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PM2.5", f"{current_data['pm2_5']:.1f} µg/m³")
        c2.metric("PM10", f"{current_data['pm10']:.1f} µg/m³")
        c3.metric("NO2", f"{current_data['nitrogen_dioxide']:.1f} µg/m³")
        c4.metric("SO2", f"{current_data['sulphur_dioxide']:.1f} µg/m³")
        c5.metric("O3", f"{current_data['ozone']:.1f} µg/m³")
        
        # Forecast
        st.write("### 3-Day Forecast")
        # Prepare feature vector (drop non-features)
        cols_to_drop = ["city", "time", "target_us_aqi_24h", "target_us_aqi_48h", "target_us_aqi_72h"]
        X_infer = features_df.drop(columns=[c for c in cols_to_drop if c in features_df.columns]).iloc[[-1]]
        X_infer = X_infer.fillna(0)
        
        preds = model.predict(X_infer)[0]
        
        # Plot Historical + Forecast
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Historical last 48h
        history = features_df.iloc[-48:]
        ax.plot(history["time"], history["us_aqi"], label="Historical AQI", color="blue")
        ax.scatter(history["time"].iloc[-1], history["us_aqi"].iloc[-1], color="blue")
        
        # Forecast points
        future_times = [
            current_data["time"] + datetime.timedelta(hours=24),
            current_data["time"] + datetime.timedelta(hours=48),
            current_data["time"] + datetime.timedelta(hours=72)
        ]
        ax.plot([history["time"].iloc[-1]] + future_times, [history["us_aqi"].iloc[-1]] + list(preds), 
                label="Forecasted AQI", color="red", linestyle="--", marker="o")
        
        ax.axhline(100, color="orange", linestyle=":", label="Unhealthy for Sensitive Grps Threshold")
        ax.set_ylabel("US AQI")
        ax.legend()
        st.pyplot(fig)
        
        # SHAP
        st.write("### Feature Importance (SHAP)")
        st.write("What's driving today's 24h forecast?")
        
        # Monkey-patch SHAP to handle XGBoost 3.0+ base_score formatting bug
        import shap.explainers._tree
        import json
        if not hasattr(shap.explainers._tree, '_patched'):
            original_decode = shap.explainers._tree.decode_ubjson_buffer
            def patched_decode(*args, **kwargs):
                jmodel = original_decode(*args, **kwargs)
                try:
                    params = jmodel["learner"]["learner_model_param"]
                    base_score = params.get("base_score")
                    if isinstance(base_score, str) and base_score.startswith("["):
                        params["base_score"] = str(json.loads(base_score)[0])
                except Exception:
                    pass
                return jmodel
            shap.explainers._tree.decode_ubjson_buffer = patched_decode
            shap.explainers._tree._patched = True

        explainer = shap.TreeExplainer(model.estimators_[0]) # first estimator is for 24h target
        shap_values = explainer.shap_values(X_infer)
        
        fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values, X_infer, plot_type="bar", show=False, max_display=10)
        st.pyplot(fig_shap)
