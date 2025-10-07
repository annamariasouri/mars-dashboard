import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os

# === Dynamic date (matches model_ready_input file name)
yesterday = datetime.today() - pd.Timedelta(days=1)
target_date_str = yesterday.strftime("%Y-%m-%d")

# === Model path
model_path = r"C:\Users\annam\OneDrive - University of Nicosia\Desktop\MARS DATA\final_rf_chl_model_2015_2023.pkl"
model = joblib.load(model_path)

# === Regions
REGIONS = ["thermaikos", "peiraeus", "limassol"]

# === Features used by the model
features = [
    'nh4','no3','po4','so','thetao',
    'chl_t-1','nh4_t-1','no3_t-1','po4_t-1','so_t-1','thetao_t-1',
    'chl_7day_avg','nh4_7day_avg','no3_7day_avg','po4_7day_avg','so_7day_avg','thetao_7day_avg',
    'n_p_ratio','n_nh4_ratio','p_nh4_ratio',
    'chl_monthly_median','chl_anomaly','bloom_proxy_label'
]

# === Loop through each region
for region in REGIONS:
    csv_input = fr"C:\Users\annam\OneDrive - University of Nicosia\Desktop\MARS DATA\model_ready_input_{region}_{target_date_str}.csv"
    output_path = fr"C:\Users\annam\OneDrive - University of Nicosia\Desktop\MARS DATA\forecast_log_{region}.csv"

    if not os.path.exists(csv_input):
        print(f"⚠️ Skipping {region} — no input file found for {target_date_str}")
        continue

    df = pd.read_csv(csv_input)

    # === Check all features
    missing = set(model.feature_names_in_) - set(df.columns)
    if missing:
        print(f"⚠️ Skipping {region} — missing required features: {missing}")
        continue

    df_input = df[model.feature_names_in_]

    # === Skip region if no valid rows
    if df_input.empty:
        print(f"⚠️ Skipping {region} — no valid rows for prediction on {target_date_str}")
        continue

    # === Predict CHL
    predicted_chl = model.predict(df_input)
    predicted_chl_mean = np.mean(predicted_chl)

    # === Dynamic threshold (per region/day)
    threshold = np.percentile(predicted_chl, 90)
    risk_flag = int(predicted_chl_mean >= threshold)

    # === Save forecast row
    result = pd.DataFrame([{
        "date": target_date_str,
        "predicted_chl": round(predicted_chl_mean, 3),
        "bloom_risk_flag": risk_flag,
        "threshold_used": round(threshold, 3),
        "num_grid_points": len(predicted_chl)
    }])

    if os.path.exists(output_path):
        result.to_csv(output_path, mode='a', index=False, header=False)
    else:
        result.to_csv(output_path, index=False)

    print(f"✅ Forecast complete for {region}:")
    print(result)
