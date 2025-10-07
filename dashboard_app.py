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

# ✅ Use current working directory for data files
data_dir = "."

st.title("🌊 MARS Dashboard – Real-Time Bloom Forecasts")

# === HELPERS ===

def latest_env_file(region: str):
    files = [f for f in os.listdir(data_dir) if re.match(rf"env_history_{region}_.+\\.csv$", f)]
    return os.path.join(data_dir, sorted(files)[-1]) if files else None


def load_forecast(region: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"forecast_log_{region}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        # coerce date column to datetime if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return pd.DataFrame()


def load_env(region: str) -> pd.DataFrame:
    f = latest_env_file(region)
    if not f:
        return pd.DataFrame()
    df = pd.read_csv(f)
    # normalize columns to UPPER for easier matching
    df.columns = [c.strip().upper() for c in df.columns]
    # common aliases
    rename_map = {"DATETIME": "TIME", "DATE": "TIME"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df


def plot_ts(df: pd.DataFrame, x: str, y: str, title: str, ylab: str):
    fig = px.line(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[COPERNICUS_BLUE])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title=ylab)
    return fig

# === MAP ===
all_lat, all_lon = [], []
for v in REGIONS.values():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    all_lat += [lat_min, lat_max]
    all_lon += [lon_min, lon_max]
center = [float(np.mean(all_lat)), float(np.mean(all_lon))]

# Default region that actually has a forecast file (fallback to thermaikos)
available = [r for r in REGIONS if os.path.exists(os.path.join(data_dir, f"forecast_log_{r}.csv"))]
default_region = available[0] if available else "thermaikos"
if "region" not in st.session_state:
    st.session_state.region = default_region

colmap = st.columns([3, 1])
with colmap[0]:
    st.subheader("📍 Regions Map (click to select)")
with colmap[1]:
    st.selectbox("Active region", options=list(REGIONS.keys()), format_func=lambda k: REGIONS[k]["title"], key="region")

m = folium.Map(location=center, zoom_start=7, tiles="cartodbpositron")
for k, v in REGIONS.items():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color=v["color"],
        fill=True,
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
region_title = REGIONS[region]["title"]

# === KPI CARDS ===
latest = forecast.tail(1)
if not latest.empty:
    row = latest.iloc[0]
    kpi = st.columns(8)
    # Access with lower-case because we normalized forecast columns
    pred_chl = row.get("predicted_chl")
    thr = row.get("threshold_used")
    bloom_flag = row.get("bloom_risk_flag")
    rec7 = row.get("recurrence_7d_prob")
    rec30 = row.get("recurrence_30d_prob")

    kpi[0].metric(f"{region_title} – CHL", f"{pred_chl:.3f} mg/m³" if pd.notna(pred_chl) else "—")
    kpi[1].metric("Bloom Flag (Today)", "Yes" if str(bloom_flag).lower() in ("1", "true", "yes") else "No")
    kpi[2].metric("Threshold Used", f"{thr:.3f}" if pd.notna(thr) else "—")
    if pd.notna(rec7):
        kpi[3].metric("Likelihood (Next 7 d)", f"{rec7} %")
    if pd.notna(rec30):
        kpi[4].metric("Likelihood (Next 30 d)", f"{rec30} %")
else:
    st.warning("No forecast data found for this region yet.")

# === TABS ===
tab1, tab2, tab3 = st.tabs(["Today’s Forecast", "Environmental Trends", "About MARS"])

with tab1:
    st.subheader(f"{region_title} – CHL Forecasts")
    if env.empty:
        st.info("No environmental history available yet.")
    else:
        # Expect a 'TIME' column and 'CHL' in env (columns set to UPPER in load_env)
        if "TIME" in env.columns:
            env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce")
            env = env.dropna(subset=["TIME"])  # keep valid timestamps
            now = env["TIME"].max()
            last7 = env[env["TIME"] >= now - timedelta(days=7)]
            last30 = env[env["TIME"] >= now - timedelta(days=30)]
            c1, c2 = st.columns(2)
            if "CHL" in env.columns:
                with c1:
                    st.plotly_chart(plot_ts(last7, "TIME", "CHL", "CHL – Last 7 days", "mg/m³"), use_container_width=True)
                with c2:
                    st.plotly_chart(plot_ts(last30, "TIME", "CHL", "CHL – Last 30 days", "mg/m³"), use_container_width=True)
        else:
            st.info("Couldn't locate a TIME column in env history.")

    if not forecast.empty and "date" in forecast.columns:
        st.plotly_chart(
            px.line(
                forecast.tail(30), x="date", y="predicted_chl",
                title="Predicted CHL (last 30 days)",
                color_discrete_sequence=[COPERNICUS_AQUA],
                template=PLOTLY_TEMPLATE,
            ),
            use_container_width=True,
        )

with tab2:
    st.subheader(f"{region_title} – Environmental Trends (30 days)")
    if env.empty:
        st.info("No env_history file found for this region yet.")
    else:
        if "TIME" in env.columns:
            env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce")
            variables = [v for v, _ in ENV_VARS if v in env.columns]
            chosen = st.multiselect("Variables to plot", variables, default=variables[:2])
            for v in chosen:
                label = dict(ENV_VARS).get(v, v)
                st.plotly_chart(plot_ts(env, "TIME", v, label, label), use_container_width=True)
        else:
            st.info("Couldn't locate a TIME column in env history.")

with tab3:
    st.markdown(
        """
        **MARS – Marine Autonomous Risk System** forecasts harmful algal bloom (red tide) risk in the
        Eastern Mediterranean using daily **Copernicus Marine** data and a trained machine-learning model.

        **Regions:** Thermaikos (GR), Piraeus (GR), Limassol (CY)  
        **Data:** NH₄, NO₃, PO₄, θ (temperature), SO (salinity), CHL  
        **Metrics shown:** Bloom flag, thresholds, risk probabilities, recurrence likelihoods  

        *Part of Annamaria Souri’s PhD Research – University of Nicosia*
        """
    )

st.markdown(
    f"<hr style='margin-top:2em;border:0;height:1px;background:{COPERNICUS_BLUE};opacity:0.3;'>"
    f"<div style='text-align:center;color:#999;font-size:12px;'>© {datetime.now().year} MARS • Research prototype</div>",
    unsafe_allow_html=True,
)
