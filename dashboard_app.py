import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

# === CONFIG ===
st.set_page_config(page_title="MARS – Marine Autonomous Risk System", page_icon="🌊", layout="wide")

COPERNICUS_BLUE = "#0072BC"
COPERNICUS_AQUA = "#00B4D8"
PLOTLY_TEMPLATE = "plotly_white"

REGIONS = {
    "thermaikos": {"title": "Thermaikos (Greece)", "bbox": (40.2, 40.7, 22.5, 23.0), "color": "#2E86DE"},
    "peiraeus": {"title": "Piraeus (Greece)", "bbox": (37.9, 38.1, 23.5, 23.8), "color": "#E17055"},
    "limassol": {"title": "Limassol (Cyprus)", "bbox": (34.6, 34.8, 33.0, 33.2), "color": "#00B894"},
}

ENV_VARS = [
    ("CHL", "Chlorophyll-a (mg/m³)"),
    ("NH4", "Ammonium NH₄ (µmol/L)"),
    ("NO3", "Nitrate NO₃ (µmol/L)"),
    ("PO4", "Phosphate PO₄ (µmol/L)"),
    ("THETAO", "Temperature θ (°C)"),
    ("SO", "Salinity (PSU)"),
]

# === SIDEBAR ===
with st.sidebar:
    st.image("https://www.copernicus.eu/sites/default/files/inline-images/Logo_Copernicus_MarineService_RGB.png", width=160)
    st.markdown("### 🌊 MARS – Marine Autonomous Risk System")
    st.write("**Annamaria Souri**, PhD Research • Powered by **Copernicus Marine**")
    data_dir = st.text_input("Data directory", ".", help="Folder containing forecast_log_*.csv and env_history_*.csv")

st.title("🌊 MARS Dashboard – Real-Time Bloom Forecasts")

# === HELPERS ===
def latest_env_file(region):
    files = [f for f in os.listdir(data_dir) if re.match(fr"env_history_{region}_.*\\.csv", f)]
    return os.path.join(data_dir, sorted(files)[-1]) if files else None

def load_forecast(region):
    path = os.path.join(data_dir, f"forecast_log_{region}.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def load_env(region):
    f = latest_env_file(region)
    return pd.read_csv(f) if f else pd.DataFrame()

def plot_ts(df, x, y, title, ylab):
    fig = px.line(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[COPERNICUS_BLUE])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title=ylab)
    return fig

# === MAP ===
all_lat, all_lon = [], []
for v in REGIONS.values():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    all_lat += [lat_min, lat_max]; all_lon += [lon_min, lon_max]
center = [np.mean(all_lat), np.mean(all_lon)]

if "region" not in st.session_state:
    st.session_state.region = "thermaikos"

colmap = st.columns([3,1])
with colmap[0]:
    st.subheader("📍 Regions Map (click to select)")
with colmap[1]:
    st.selectbox("Active region", options=list(REGIONS.keys()), format_func=lambda k: REGIONS[k]["title"], key="region")

m = folium.Map(location=center, zoom_start=7, tiles="cartodbpositron")
for k, v in REGIONS.items():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color=v["color"], fill=True,
        fill_opacity=0.25 if k != st.session_state.region else 0.5,
        popup=v["title"],
    ).add_to(m)

click = st_folium(m, height=420, key="mars_map")
if click and click.get("last_clicked"):
    lat, lon = click["last_clicked"]["lat"], click["last_clicked"]["lng"]
    for k, v in REGIONS.items():
        lat_min, lat_max, lon_min, lon_max = v["bbox"]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            st.session_state.region = k

region = st.session_state.region
forecast = load_forecast(region)
env = load_env(region)
title = REGIONS[region]["title"]

# === KPI CARDS ===
latest = forecast.tail(1)
if not latest.empty:
    row = latest.iloc[0]
    kpi = st.columns(8)
    kpi[0].metric(f"{title} – CHL", f"{row['predicted_chl']:.3f} mg/m³")
    kpi[1].metric("Bloom Flag (Today)", "Yes" if row['bloom_risk_flag']==1 else "No")
    kpi[2].metric("Threshold Used", f"{row['threshold_used']:.3f}")
    if "recurrence_7d_prob" in row:
        kpi[3].metric("Likelihood (Next 7 d)", f"{row['recurrence_7d_prob']} %")
    if "recurrence_30d_prob" in row:
        kpi[4].metric("Likelihood (Next 30 d)", f"{row['recurrence_30d_prob']} %")
else:
    st.warning("No forecast data found for this region yet.")

# === TABS ===
tab1, tab2, tab3 = st.tabs(["Today’s Forecast", "Environmental Trends", "About MARS"])

with tab1:
    st.subheader(f"{title} – CHL Forecasts")
    if env.empty:
        st.info("No environmental history available yet.")
    else:
        env["time"] = pd.to_datetime(env["time"], errors="coerce")
        env = env.dropna(subset=["time"])
        now = env["time"].max()
        seven = env[env["time"] >= now - timedelta(days=7)]
        thirty = env[env["time"] >= now - timedelta(days=30)]
        c1, c2 = st.columns(2)
        if "CHL" in env.columns:
            with c1: st.plotly_chart(plot_ts(seven, "time", "CHL", "CHL – Last 7 days", "mg/m³"), use_container_width=True)
            with c2: st.plotly_chart(plot_ts(thirty, "time", "CHL", "CHL – Last 30 days", "mg/m³"), use_container_width=True)
    if not forecast.empty:
        st.plotly_chart(px.line(forecast.tail(30), x="date", y="predicted_chl",
                                title="Predicted CHL (last 30 days)", color_discrete_sequence=[COPERNICUS_AQUA],
                                template=PLOTLY_TEMPLATE), use_container_width=True)

with tab2:
    st.subheader(f"{title} – Environmental Trends (30 days)")
    if env.empty:
        st.info("No env_history file found for this region yet.")
    else:
        env["time"] = pd.to_datetime(env["time"], errors="coerce")
        variables = [v for v,_ in ENV_VARS if v in env.columns]
        chosen = st.multiselect("Variables to plot", variables, default=variables[:2])
        for v in chosen:
            label = dict(ENV_VARS).get(v, v)
            st.plotly_chart(plot_ts(env, "time", v, label, label), use_container_width=True)

with tab3:
    st.markdown("""
    **MARS – Marine Autonomous Risk System** forecasts harmful algal bloom (red tide) risk in the
    Eastern Mediterranean using daily **Copernicus Marine** data and a trained machine-learning model.

    **Regions:** Thermaikos (GR), Piraeus (GR), Limassol (CY)  
    **Data:** NH₄, NO₃, PO₄, θ (temperature), SO (salinity), CHL  
    **Metrics shown:** Bloom flag, thresholds, risk probabilities, recurrence likelihoods  

    *Part of Annamaria Souri’s PhD Research – University of Nicosia*
    """)

st.markdown(f"<hr style='margin-top:2em;border:0;height:1px;background:{COPERNICUS_BLUE};opacity:0.3;'>"
            f"<div style='text-align:center;color:#999;font-size:12px;'>© {datetime.now().year} MARS • Research prototype</div>",
            unsafe_allow_html=True)
