# MARS (Marine Autonomous Risk System) – Streamlit Dashboard
# Author: Annamaria Souri
# Notes:
# Fix for indentation error and final tab initialization.

import os
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="MARS – Marine Autonomous Risk System", page_icon="🌊", layout="wide")

DEFAULT_DATA_DIR = os.environ.get("MARS_DATA_DIR", ".")
REGIONS = {
    "thermaikos": {"title": "Thermaikos (Greece)", "bbox": (40.2, 40.7, 22.5, 23.0), "color": "#2E86DE"},
    "piraeus": {"title": "Piraeus (Greece)", "bbox": (37.9, 38.1, 23.5, 23.8), "color": "#E17055"},
    "limassol": {"title": "Limassol (Cyprus)", "bbox": (34.6, 34.8, 33.0, 33.2), "color": "#00B894"},
}

@st.cache_data(ttl=3600)
def load_data(region, base_dir):
    forecast_path = Path(base_dir) / f"forecast_log_{region}.csv"
    env_pattern = list(Path(base_dir).glob(f"env_history_{region}_*.csv"))
    env_path = max(env_pattern, default=None, key=os.path.getmtime)
    forecast = pd.read_csv(forecast_path) if forecast_path.exists() else pd.DataFrame()
    env = pd.read_csv(env_path) if env_path else pd.DataFrame()
    return forecast, env

# Sidebar
with st.sidebar:
    st.markdown("### 🌊 MARS – Marine Autonomous Risk System")
    st.write("Part of **Annamaria Souri**’s PhD research • Powered by **Copernicus Marine**")
    data_dir = st.text_input("Data directory", value=str(DEFAULT_DATA_DIR))

# Title
st.title("MARS – Marine Autonomous Risk System")

# Map
all_lats = [v for r in REGIONS.values() for v in (r['bbox'][0], r['bbox'][1])]
all_lons = [v for r in REGIONS.values() for v in (r['bbox'][2], r['bbox'][3])]
center_lat, center_lon = np.mean(all_lats), np.mean(all_lons)

region_keys = list(REGIONS.keys())
if "selected_region" not in st.session_state:
    st.session_state.selected_region = region_keys[0]

fol_map = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="cartodbpositron")
for key, meta in REGIONS.items():
    lat_min, lat_max, lon_min, lon_max = meta["bbox"]
    coords = [[lat_min, lon_min], [lat_min, lon_max], [lat_max, lon_max], [lat_max, lon_min], [lat_min, lon_min]]
    folium.Polygon(
        [[c[0], c[1]] for c in coords],
        color=meta["color"],
        fill=True,
        fill_opacity=0.3 if key == st.session_state.selected_region else 0.1,
        tooltip=meta["title"],
    ).add_to(fol_map)

mret = st_folium(fol_map, height=450, key="map")
if mret and mret.get("last_clicked"):
    lat, lon = mret["last_clicked"]["lat"], mret["last_clicked"]["lng"]
    for k, v in REGIONS.items():
        lat_min, lat_max, lon_min, lon_max = v["bbox"]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            st.session_state.selected_region = k

region = st.session_state.selected_region
forecast_df, env_df = load_data(region, data_dir)

# Tabs
t1, t2, t3 = st.tabs(["Today’s Forecast", "Environmental Trends", "About the MARS System"])

with t1:
    st.subheader(f"{REGIONS[region]['title']} – Today’s Forecast")
    if forecast_df.empty:
        st.info("No forecast data found.")
    else:
        st.dataframe(forecast_df.tail())

with t2:
    st.subheader(f"{REGIONS[region]['title']} – Environmental Trends")
    if env_df.empty:
        st.info("No environmental data found.")
    else:
        st.dataframe(env_df.tail())

with t3:
    st.markdown("**MARS** is a prototype dashboard using Copernicus Marine data and ML models to forecast red tide risk.")

st.markdown(
    f"<div style='text-align:center;color:#999;font-size:12px;margin-top:24px;'>© {datetime.now().year} MARS • Research prototype</div>",
    unsafe_allow_html=True,
)

