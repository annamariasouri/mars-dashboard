import copernicusmarine
import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# === Parameters ===
yesterday = datetime.today() - timedelta(days=1)
target_date = yesterday.strftime("%Y-%m-%dT00:00:00")
target_date_str = yesterday.strftime("%Y-%m-%d")

REGIONS = {
    "thermaikos": (40.2, 40.7, 22.5, 23.0),
    "peiraeus": (37.9, 38.1, 23.5, 23.8),
    "limassol": (34.6, 34.8, 33.0, 33.2)
}

output_dir = r"C:\Users\annam\OneDrive - University of Nicosia\Desktop\MARS DATA"
os.makedirs(output_dir, exist_ok=True)

# === Authenticate ===
copernicusmarine.login()

# === Dataset list ===
datasets = [
    ("cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m", ["nh4", "no3", "po4"]),
    ("cmems_mod_med_phy-tem_anfc_4.2km_P1D-m", ["thetao"]),
    ("cmems_mod_med_phy-sal_anfc_4.2km_P1D-m", ["so"]),
    ("cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m", ["chl"])
]

# === Process each region ===
for region, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
    print(f"\n🌍 Processing region: {region.upper()}")

    region_dir = os.path.join(output_dir, f"{region}_downloads_{target_date_str}")
    os.makedirs(region_dir, exist_ok=True)
    os.chdir(region_dir)

    print("📥 Starting downloads...")
    for dataset_id, vars in datasets:
        print(f"→ Downloading {vars} from {dataset_id}")
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=vars,
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=(yesterday - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00"),
            end_datetime=target_date,
            minimum_depth=1.0,
            maximum_depth=5.0
        )

    # === Collect the most recent .nc files in this folder
    nc_files = sorted([f for f in os.listdir(region_dir) if f.endswith(".nc")], key=os.path.getctime, reverse=True)
    if len(nc_files) < 4:
        print(f"⚠️ Not all datasets were downloaded for {region}. Found {len(nc_files)} files.")
        continue

    chl_file, sal_file, temp_file, nut_file = nc_files[0], nc_files[1], nc_files[2], nc_files[3]

    df_chl = xr.open_dataset(chl_file).to_dataframe().reset_index()
    df_sal = xr.open_dataset(sal_file).to_dataframe().reset_index()
    df_temp = xr.open_dataset(temp_file).to_dataframe().reset_index()
    df_nut = xr.open_dataset(nut_file).to_dataframe().reset_index()

    # === Merge datasets
    merge_keys = ['time', 'depth', 'latitude', 'longitude']
    df = df_nut.merge(df_temp, on=merge_keys, how='outer')
    df = df.merge(df_sal, on=merge_keys, how='outer')
    df = df.merge(df_chl, on=merge_keys, how='outer')
    df = df.sort_values(by=["latitude", "longitude", "time"]).reset_index(drop=True)

    # === Save full 30-day environmental history (before feature engineering)
    env_cols = ["time", "chl", "nh4", "no3", "po4", "thetao", "so"]
    df_env = df[env_cols].copy()
    df_env = df_env.groupby("time").mean().reset_index()  # daily average
    history_filename = f"env_history_{region}_{target_date_str}.csv"
    df_env.to_csv(os.path.join(output_dir, history_filename), index=False)
    print(f"📤 Saved environmental history: {history_filename}")

    # === Feature engineering
    group = df.groupby(['latitude', 'longitude'])

    def add_lag_and_rolling(df, var):
        df[f"{var}_t-1"] = group[var].shift(1)
        df[f"{var}_7day_avg"] = group[var].rolling(window=7, min_periods=1).mean().reset_index(drop=True)

    for var in ["chl", "nh4", "no3", "po4", "so", "thetao"]:
        add_lag_and_rolling(df, var)

    df["chl_monthly_median"] = group["chl"].rolling(window=30, min_periods=1).median().reset_index(drop=True)
    df["chl_anomaly"] = df["chl"] - df["chl_monthly_median"]

    df["n_p_ratio"] = np.where(df["po4"] != 0, df["no3"] / df["po4"], np.nan)
    df["n_nh4_ratio"] = np.where(df["nh4"] != 0, df["no3"] / df["nh4"], np.nan)
    df["p_nh4_ratio"] = np.where(df["nh4"] != 0, df["po4"] / df["nh4"], np.nan)

    df["bloom_proxy_label"] = 0

    # === Keep only final date's data
    df_latest = df[df["time"] == df["time"].max()]

    model_features = [
        'nh4','no3','po4','so','thetao',
        'chl_t-1','nh4_t-1','no3_t-1','po4_t-1','so_t-1','thetao_t-1',
        'chl_7day_avg','nh4_7day_avg','no3_7day_avg','po4_7day_avg','so_7day_avg','thetao_7day_avg',
        'n_p_ratio','n_nh4_ratio','p_nh4_ratio',
        'chl_monthly_median','chl_anomaly','bloom_proxy_label'
    ]

    df_ready = df_latest.dropna(subset=model_features)[model_features]
    filename = f"model_ready_input_{region}_{target_date_str}.csv"
    final_path = os.path.join(output_dir, filename)
    df_ready.to_csv(final_path, index=False)

    print(f"✅ Saved model input: {final_path}")
    print(df_ready.head())
