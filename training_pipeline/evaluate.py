import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

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

def get_aqi_category(aqi: float) -> str:
    """Standard US AQI Breakpoints"""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"
