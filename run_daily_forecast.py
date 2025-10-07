import pandas as pd
import joblib
import numpy as np
import io
import requests
from datetime import datetime, timedelta
import os

# === Parameters ===
yesterday = datetime.today() - timedelta(days=1)
target_date_str = yesterday.strftime("%Y-%m-%d")
data_dir = os.environ.get("MARS_DATA_DIR", ".")  # works locally and on GitHub Actions

# === Load model from GitHub Release ===
print("📦 Downloading trained model from GitHub release...")
model_url = "https://github.com/annamariasouri/mars-dashboard/releases/download/v1.0/final_rf_chl_model_2015_2023.pkl"
response = requests.get(model_url)
response.raise_for_status()
model = joblib.load(io.BytesIO(response.content))
print("✅ Model loaded successfully!")

# === Regions ===
REGIONS = ["thermaikos", "peiraeus", "limassol"]

# === Features used by the model ===
features = [
    'nh4','no3','po4','so','thetao',
    'chl_t-1','nh4_t-1','no3_t-1','po4_t-1','so_t-1','thetao_t-1',
    'chl_7day_avg','nh4_7day_avg','no3_7day_avg','po4_7day_avg','so_7day_avg','thetao_7day_avg',
    'n_p_ratio','n_nh4_ratio','p_nh4_ratio',
    'chl_monthly_median','chl_anomaly','bloom_proxy_label'
]

# === Loop through each region ===
for region in REGIONS:
    csv_input = os.path.join(data_dir, f"model_ready_input_{region}_{target_date_str}.csv")
    output_path = os.path.join(data_dir, f"forecast_log_{region}.csv")

    if not os.path.exists(csv_input):
        print(f"⚠️ Skipping {region} — no input file found for {target_date_str}")
        continue

    df = pd.read_csv(csv_input)

    # === Check features ===
    missing = set(model.feature_names_in_) - set(df.columns)
    if missing:
        print(f"⚠️ Skipping {region} — missing required features: {missing}")
        continue

    df_input = df[model.feature_names_in_]
    if df_input.empty:
        print(f"⚠️ Skipping {region} — no valid rows for prediction on {target_date_str}")
        continue

    # === Predict CHL ===
    predicted_chl = model.predict(df_input)
    predicted_chl_mean = np.mean(predicted_chl)

    # === Dynamic threshold (90th percentile per region/day)
    threshold = np.percentile(predicted_chl, 90)
    risk_flag = int(predicted_chl_mean >= threshold)

    # === Compute recurrence likelihoods ===
    # Use rolling averages of previous forecasts for 7- and 30-day context
    recurrence_7d = None
    recurrence_30d = None

    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        if "predicted_chl" in prev.columns and "threshold_used" in prev.columns:
            prev = prev.tail(30)  # last 30 days if available
            prev["risk_flag"] = prev["predicted_chl"] >= prev["threshold_used"]
            recurrence_7d = round(prev["risk_flag"].tail(7).mean() * 100, 1) if len(prev) >= 1 else None
            recurrence_30d = round(prev["risk_flag"].mean() * 100, 1)
    else:
        recurrence_7d = np.nan
        recurrence_30d = np.nan

    # === Save forecast row ===
    result = pd.DataFrame([{
        "date": target_date_str,
        "predicted_chl": round(predicted_chl_mean, 3),
        "bloom_risk_flag": risk_flag,
        "threshold_used": round(threshold, 3),
        "num_grid_points": len(predicted_chl),
        "recurrence_7d_prob": recurrence_7d,
        "recurrence_30d_prob": recurrence_30d
    }])

    # Append or create new forecast log
    if os.path.exists(output_path):
        result.to_csv(output_path, mode='a', index=False, header=False)
    else:
        result.to_csv(output_path, index=False)

    print(f"✅ Forecast complete for {region}:")
    print(result)
