import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import os

# === Region definitions ===
REGIONS = {
    "thermaikos": {"name": "Thermaikos", "lat": 40.5, "lon": 22.75},
    "peiraeus": {"name": "Piraeus", "lat": 37.95, "lon": 23.65},
    "limassol": {"name": "Limassol", "lat": 34.67, "lon": 33.03}
}

# === Load forecast data for each region ===
def load_forecast(region):
    path = f"forecast_log_{region}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date")
        return df
    else:
        return None

# === Page layout ===
st.set_page_config(page_title="MARS Forecast Map", layout="wide")
st.title("🌊 MARS Red Tide Risk Map")
st.caption("Copernicus-powered forecasting across the Eastern Mediterranean")

# === CHL / Red Tide Explanation ===
with st.expander("ℹ️ What is CHL and Why It Matters?", expanded=True):
    st.markdown("""
    **Chlorophyll-a (CHL)** is a proxy for phytoplankton levels in the ocean.  
    When CHL is abnormally high, it can signal **algal blooms**, including harmful red tides.

    Red tide events can:
    - Release toxins harmful to marine life and humans
    - Deplete oxygen and
