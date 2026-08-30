# AQI Predictor: Pakistan AQI 3-Day Forecast

An end-to-end machine learning system that predicts the Air Quality Index (AQI) for four major Pakistani cities using XGBoost, Open-Meteo data, and Hopsworks as the feature store and model registry, served through an interactive Streamlit dashboard with SHAP explainability.

**Live Demo:** [https://aqi-predictor-oapp6n5ejnccchssu3krbqu.streamlit.app/](https://aqi-predictor-oapp6n5ejnccchssu3krbqu.streamlit.app/)

## Setup & Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

> [!WARNING]
> **Windows Installation Issue (`twofish` dependency):**
> The `hopsworks` package depends on `pyjks`, which in turn requires the `twofish` cryptography package. `twofish` has C extensions that require Microsoft Visual C++ Build Tools to compile on Windows. If your installation fails with a C-compiler error, you must either install the MSVC build tools, or run the project in a WSL (Windows Subsystem for Linux) environment where GCC is available. Alternatively, you can use Hopsworks' built-in Jupyter environment which already has the dependencies resolved.

## System Architecture

```mermaid
graph LR
    A["Open-Meteo API"] -->|Weather + AQI| B["Feature Pipeline"]
    B -->|Engineered Features| C["Hopsworks Feature Store"]
    C -->|Training Data| D["Training Pipeline"]
    D -->|Registered Model| E["Hopsworks Model Registry"]
    C -->|Live Features| F["Streamlit Dashboard"]
    E -->|XGBoost Model| F
    F -->|Predictions + SHAP| G["User Browser"]
```

| Component | Technology | Location |
|---|---|---|
| Data Source | Open-Meteo (Weather + Air Quality APIs) | External API |
| Feature Store | Hopsworks (Feature Group + Feature View) | Cloud (EU-West) |
| Model Registry | Hopsworks Model Registry | Cloud (EU-West) |
| Historical Backfill | Self-contained Python script | Hopsworks Jupyter |
| Model Training | Self-contained Python script | Hopsworks Jupyter |
| Dashboard | Streamlit | Local / Deployable |
| CI/CD | GitHub Actions (manual trigger only) | GitHub |
| Local Backup | Per-city CSV exports | `local_backup/` |

## Repository Structure

```
AQI Predictor/
├── app/
│   └── app.py                    # Streamlit dashboard
├── feature_pipeline/
│   ├── config.py                 # Central config, API keys, schema definition
│   ├── fetch_data.py             # Open-Meteo API data fetching with retries
│   ├── features.py               # Feature engineering (40+ features)
│   ├── backfill.py               # GitHub Actions backfill (deprecated)
│   └── pipeline.py               # Hourly incremental pipeline
├── training_pipeline/
│   ├── train.py                  # Model training (GitHub Actions version)
│   └── evaluate.py               # Evaluation metrics + AQI category mapping
├── .github/workflows/
│   ├── feature_pipeline.yml      # Hourly feature ingestion (manual only)
│   └── training_pipeline.yml     # Daily training (manual only)
├── hopsworks_backfill.py         # Self-contained backfill for Hopsworks Jupyter
├── hopsworks_train.py            # Self-contained training for Hopsworks Jupyter
├── export_existing_data.py       # Export feature store data to local CSV
├── analyze_local.py              # Debug utility: Test model training locally from CSVs
├── analyze_model.py              # Debug utility: Extract metrics/feature importance from Hopsworks model
├── fix_schema.py                 # Debug utility: Script to recreate/fix feature group schema
├── test_shap.py                  # Debug utility: Test SHAP explainer locally
├── requirements.txt              # Python dependencies
└── .env.example                  # Template for API keys
```

## Data Pipeline

### Data Sources

All data comes from the free [Open-Meteo API](https://open-meteo.com/) (no API key required):

| Endpoint | Variables |
|---|---|
| **Weather (Historical)** `archive-api.open-meteo.com` | temperature\_2m, relative\_humidity\_2m, wind\_speed\_10m, wind\_direction\_10m, precipitation, surface\_pressure, cloud\_cover |
| **Weather (Forecast)** `api.open-meteo.com` | Same 7 variables (used for live inference) |
| **Air Quality** `air-quality-api.open-meteo.com` | pm10, pm2\_5, carbon\_monoxide, nitrogen\_dioxide, sulphur\_dioxide, ozone, us\_aqi |

### Cities

| City | Latitude | Longitude |
|---|---|---|
| Islamabad | 33.6844 | 73.0479 |
| Karachi | 24.8607 | 67.0011 |
| Hyderabad | 25.3960 | 68.3578 |
| Lahore | 31.5204 | 74.3587 |

### Historical Backfill

Two years of hourly data (~15,800 rows per city, ~63,000 total) was ingested using `hopsworks_backfill.py`, run inside Hopsworks' own Jupyter environment. Key design decisions:

- **Monthly chunking**: Data is inserted in 30-day chunks to avoid large write timeouts
- **Resume-safe**: Skips fully backfilled cities and already-inserted chunks using `(city, time)` primary key lookups
- **Retry logic**: Exponential backoff (3 attempts, 10s/20s/40s delays) on all Open-Meteo API calls
- **Local backup**: Every inserted chunk is also saved as a CSV to `local_backup/`

### Feature Store Schema

The feature group `aqi_features` (version 1) uses an explicitly defined schema with 48 columns via `hsfs.feature.Feature` objects to prevent dynamic inference mismatches. All integer-derived columns use `bigint` (int64). Primary key is `(city, time)` with `time` as event time.

## Feature Engineering

`features.py` computes **40+ features** from the raw 14 input variables:

| Category | Count | Features |
|---|---|---|
| **Time-Based** | 5 | `hour`, `day_of_week`, `month`, `is_weekend`, `season` |
| **Lag Features** | 10 | US AQI and PM2.5 at t-1, t-3, t-6, t-24, t-48 hours |
| **Rate of Change** | 1 | `us_aqi_diff_1` (current AQI minus 1-hour-ago AQI) |
| **Rolling Windows** | 15 | Min, Max, Mean, Std of US AQI over 6h, 24h, 168h windows; Mean of PM2.5 over same windows |
| **Raw Inputs** | 14 | 7 weather + 7 air quality variables passed through |
| **Targets** | 3 | `target_us_aqi_24h`, `target_us_aqi_48h`, `target_us_aqi_72h` |

## Model Training and Evaluation

Three models are trained and compared using a 20% random split on the historical feature view (63,360 total rows):

- **Training set:** 50,688 rows (80%)
- **Testing set:** 12,672 rows (20%)

| Model | Configuration |
|---|---|
| **Ridge Regression** (baseline) | `alpha=1.0`, wrapped in `MultiOutputRegressor` |
| **LSTM** (deep learning) | `Sequential([LSTM(64), Dense(32), Dense(3)])`, trained for 10 epochs (adam, mse) |
| **XGBoost** (production) | `n_estimators=100, learning_rate=0.1, max_depth=5`, wrapped in `MultiOutputRegressor` |

### Results

| Metric | Ridge (24h) | LSTM (24h) | XGBoost (24h) | Ridge (48h) | LSTM (48h) | XGBoost (48h) | Ridge (72h) | LSTM (72h) | XGBoost (72h) |
|---|---|---|---|---|---|---|---|---|---|
| **RMSE** | 16.88 | 16.07 | **13.41** | 22.84 | 20.77 | **15.91** | 24.50 | 22.68 | **16.05** |
| **MAE** | 11.54 | 11.21 | **9.55** | 16.19 | 14.64 | **11.79** | 17.70 | 16.10 | **12.05** |
| **R2** | 0.81 | 0.83 | **0.88** | 0.64 | 0.70 | **0.83** | 0.58 | 0.64 | **0.82** |

As expected for tabular data, **XGBoost outperforms both the Ridge baseline and the LSTM neural network** across all horizons by a wide margin. The deep learning LSTM model successfully bridges the gap between the linear baseline and the tree-based model, but cannot match XGBoost's predictive power on this structured dataset.

Notably, XGBoost's 72-hour R2 of **0.82** essentially matches the LSTM's 24-hour R2 of **0.83**, demonstrating XGBoost's superior forecasting stability 3 days out.

### Feature Importance

Averaging the feature importances across the 3 XGBoost estimators (24h, 48h, 72h targets), the top 5 most predictive features are:
1. `us_aqi` (Current AQI): ~48% importance
2. `us_aqi_roll_mean_168` (7-day rolling mean AQI): ~12% importance
3. `pm2_5_roll_mean_24` (24h rolling mean PM2.5): ~7% importance
4. `pm2_5_roll_mean_168` (7-day rolling mean PM2.5): ~5% importance
5. `pm2_5` (Current PM2.5): ~2% importance

*(Note: The AQI categories like "Good", "Moderate", etc., are deterministic mappings based on standard US EPA breakpoints, not a separate trained classifier model. The model directly regresses the raw AQI value and the dashboard maps it to a category.)*

The XGBoost model is registered in the Hopsworks Model Registry as `aqi_xgboost_multi` (version 1) with full input/output schema metadata.

## Dashboard

The Streamlit dashboard (`app/app.py`) provides five main components:

1. **City Selector** (sidebar): Dropdown to switch between Islamabad, Karachi, Hyderabad, and Lahore
2. **Hazard Banner**: Color-coded banner showing current AQI value and category using standard US AQI breakpoints
3. **Current Pollutants**: Five metric cards displaying live PM2.5, PM10, NO2, SO2, and O3 concentrations
4. **3-Day Forecast Chart**: Historical AQI (last 48h) plus forecasted 24h/48h/72h AQI with an unhealthy threshold line
5. **SHAP Explainability**: Bar chart showing the top 10 features driving the 24-hour forecast

### AQI Category Mapping

| AQI Range | Category | Banner Color |
|---|---|---|
| 0 to 50 | Good | Green |
| 51 to 100 | Moderate | Yellow |
| 101 to 150 | Unhealthy for Sensitive Groups | Orange |
| 151 to 200 | Unhealthy | Red |
| 201 to 300 | Very Unhealthy | Purple |
| 301+ | Hazardous | Maroon |

## How to Run

### Prerequisites

- Python 3.11+
- Hopsworks account with a configured project
- `.env` file with `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT_NAME` (see `.env.example`)

### Local Dashboard

```bash
# Activate virtual environment
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/app.py
```

### Historical Backfill (inside Hopsworks Jupyter)

1. Open Hopsworks UI, go to Jupyter from the sidebar
2. Paste the contents of `hopsworks_backfill.py` into a notebook cell
3. Run the cell (resume-safe; can be re-run without duplicating data)

### Model Training (inside Hopsworks Jupyter)

1. Open Hopsworks UI, go to Jupyter from the sidebar
2. Paste the contents of `hopsworks_train.py` into a notebook cell
3. Run the cell (creates feature view, trains models, registers the best model)

### Export Data Backup

```bash
python export_existing_data.py
```

Saves all feature store data as per-city CSVs to `local_backup/`.

## Challenges Overcome

| Challenge | Root Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: feature_pipeline` | GitHub Actions missing PYTHONPATH | Added `PYTHONPATH: .` to workflow YAML |
| Open-Meteo `ReadTimeoutError` | Transient network timeout | 120s timeout + 3-attempt exponential backoff |
| Hopsworks schema mismatch | Auto-inferred schema had inconsistent int types | Explicit `Feature()` schema definition, all integers as `bigint` |
| HDFS `RPC listener disconnected` | Network incompatibility between GH Actions and Hopsworks HDFS | Moved backfill/training to Hopsworks Jupyter |
| `train_test_split()` on `NoneType` | `get_feature_view()` returns `None` instead of raising | Added explicit `if feature_view is None: raise` check |
| `squared=False` TypeError | Removed in scikit-learn 1.4+ | Replaced with `np.sqrt(mean_squared_error(...))` |
| SHAP `ValueError` on base\_score | XGBoost 3.0+ serializes base\_score as JSON array | Monkey-patched `decode_ubjson_buffer` to parse the array |
| `fillna(method='ffill')` warning | Deprecated in pandas 2.1+ | Replaced with `.ffill()` |

## Future Work

- **Incremental Updates**: Set up a lightweight cron inside Hopsworks to keep features fresh
- **Multi-City Comparison**: Side-by-side AQI comparison view across all four cities
- **Alerts**: Telegram or email notifications when AQI crosses unhealthy thresholds
- **Anomaly Detection**: Flag unusual AQI spikes that deviate from model predictions
- **Health Advisories**: Contextual health recommendations based on AQI category
- **Model Retraining**: Automated periodic retraining as more data accumulates
