# AQI Predictor: Pakistan AQI 3-Day Forecast

An end-to-end machine learning system that predicts the Air Quality Index (AQI) for four major Pakistani cities using XGBoost, Open-Meteo data, and Hopsworks as the feature store and model registry, served through an interactive Streamlit dashboard with SHAP explainability.

**Live Demo:** [https://aqi-predictor-oapp6n5ejnccchssu3krbqu.streamlit.app/](https://aqi-predictor-oapp6n5ejnccchssu3krbqu.streamlit.app/)

**Full Project Report:** [Project_Report.pdf](./Project_Report.pdf)

## Setup & Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

> [!WARNING]
> **Windows Installation Issue (`twofish` dependency):**
> The `hopsworks` package depends on `pyjks`, which in turn requires the `twofish` cryptography package. `twofish` has C extensions that require Microsoft Visual C++ Build Tools to compile on Windows. If your installation fails with a C-compiler error, you must either install the MSVC build tools, or run the project in a WSL (Windows Subsystem for Linux) environment where GCC is available. Alternatively, you can use Hopsworks' built-in Jupyter environment which already has the dependencies resolved.

## Prerequisites

- Python 3.11+
- Hopsworks account with a configured project
- `.env` file with `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT_NAME` (see `.env.example`)

## How to Run

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

## Repository Structure

```
AQI Predictor/
├── app/
│   └── app.py                    # Streamlit dashboard
├── feature_pipeline/
│   ├── config.py                 # Central config, API keys, schema definition
│   ├── fetch_data.py             # Open-Meteo API data fetching with retries
│   ├── features.py               # Feature engineering
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
├── Project_Report.pdf            # Detailed project report
├── requirements.txt              # Python dependencies
└── .env.example                  # Template for API keys
```


For full details on the architecture, data pipeline, feature engineering, model comparison, dashboard, and challenges faced during development, see **[Project_Report.pdf](./Project_Report.pdf)**.
