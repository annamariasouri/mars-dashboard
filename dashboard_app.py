# MARS (Marine Autonomous Risk System) – Streamlit Dashboard
# Author: Annamaria Souri
# Notes:
# - Expects daily CSVs prepared by your ETL pipeline:
#     forecast_log_<region>.csv                 # daily predictions with risk probabilities
#     env_history_<region>_<YYYY-MM-DD>.csv     # rolling 30-day env. variables
# - Regions supported: thermaikos, piraeus, limassol
# - Map is interactive (click polygons) via streamlit-folium
# - Install deps: pip install streamlit pandas numpy plotly folium streamlit-folium pyproj
# - Run: streamlit run app.py

import os
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer
import folium
from streamlit_folium import st_folium

# =========================
# ---- CONFIG / CONSTANTS
# =========================
st.set_page_config(
    page_title="MARS – Marine Autonomous Risk System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Root folder where your daily files are stored (can be a mounted drive or local)
# You can override from Streamlit sidebar.
DEFAULT_DATA_DIR = os.environ.get("MARS_DATA_DIR", ".")

# Region metadata: names, short codes, polygons (lat, lon), map centroids
REGIONS = {
    "thermaikos": {
        "title": "Thermaikos (Greece)",
        "bbox": (40.2, 40.7, 22.5, 23.0),  # (lat_min, lat_max, lon_min, lon_max)
        "color": "#2E86DE",
    },
    "piraeus": {
        "title": "Piraeus (Greece)",
        "bbox": (37.9, 38.1, 23.5, 23.8),
        "color": "#E17055",
    },
    "limassol": {
        "title": "Limassol (Cyprus)",
        "bbox": (34.6, 34.8, 33.0, 33.2),
        "color": "#00B894",
    },
}

# Variables to chart
ENV_VARS = [
    ("CHL", "Chlorophyll-a (mg/m³)"),
    ("NH4", "Ammonium NH₄ (µmol/L)"),
    ("NO3", "Nitrate NO₃ (µmol/L)"),
    ("PO4", "Phosphate PO₄ (µmol/L)"),
    ("SST", "Sea Surface Temperature θ (°C)"),
    ("SAL", "Salinity (PSU)"),
]

# Preferred Plotly template for a clean, professional look
PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#0F4C81"  # professional blue accent

# =========================
# ---- HELPERS
# =========================

def find_latest_env_history(data_dir: Path, region: str) -> Path | None:
    """Return the path of the latest env_history CSV for a region, else None."""
    # Pattern: env_history_<region>_<date>.csv
    pattern = re.compile(rf"env_history_{region}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
    latest_date = None
    latest_path = None
    for p in data_dir.glob(f"env_history_{region}_*.csv"):
        m = pattern.search(p.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d")
        if (latest_date is None) or (d > latest_date):
            latest_date = d
            latest_path = p
    return latest_path


def load_forecast_log(data_dir: Path, region: str) -> pd.DataFrame:
    path = data_dir / f"forecast_log_{region}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Expected (flexible) columns: date, risk_prob, bloom_flag, threshold, CHL_pred, ...
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Parse date
    for col in ("date", "timestamp", "ds"):
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col])
            break
    if "date" not in df.columns:
        # try index as date
        df["date"] = pd.to_datetime(df.index)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_env_history(data_dir: Path, region: str) -> pd.DataFrame:
    path = find_latest_env_history(data_dir, region)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    # Expect at least a datetime column – try common options
    time_col = None
    for c in ["DATE", "DATETIME", "TIME", "TS"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # If missing, try to reconstruct from index
        df["DATETIME"] = pd.to_datetime(df.index)
        time_col = "DATETIME"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: "DATETIME"}).sort_values("DATETIME").reset_index(drop=True)

    # Harmonize env variable names
    rename_map = {"THETA": "SST", "T": "SST", "TEMP": "SST", "SO": "SAL"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def summarize_region(forecast: pd.DataFrame, env: pd.DataFrame) -> dict:
    """Compute cards & metrics for a region."""
    summary = {
        "latest_chl": None,
        "bloom_flag": None,
        "threshold": None,
        "risk_prob": None,
        "risk_days_7": None,
        "risk_days_30": None,
        "recurrence_7d_prob": None,
        "last_update": None,
    }

    if not forecast.empty:
        last_row = forecast.iloc[-1]
        summary["risk_prob"] = float(last_row.get("risk_prob", np.nan)) if "risk_prob" in forecast.columns else np.nan
        bf = last_row.get("bloom_flag")
        if isinstance(bf, str):
            summary["bloom_flag"] = bf.lower() in ("1", "true", "yes", "y")
        else:
            summary["bloom_flag"] = bool(bf) if bf is not None and not pd.isna(bf) else None
        summary["threshold"] = last_row.get("threshold")
        # Some pipelines include CHL prediction or observed CHL
        chl_col = None
        for cand in ["chl_pred", "chl", "chl_mg_m3", "chlorophyll", "chl_prediction"]:
            if cand in forecast.columns:
                chl_col = cand
                break
        if chl_col:
            try:
                summary["latest_chl"] = float(last_row[chl_col])
            except Exception:
                summary["latest_chl"] = None

        # Risk-day counts
        # Identify a boolean bloom flag vector best as possible
        b = None
        for cand in ["bloom_flag", "bloom", "risk_flag", "label_pred"]:
            if cand in forecast.columns:
                tmp = forecast[cand]
                if tmp.dtype == bool:
                    b = tmp
                else:
                    b = tmp.astype(str).str.lower().isin(["1", "true", "yes", "y"])
                break
        if b is not None:
            cutoff_7 = forecast["date"].max() - timedelta(days=7)
            cutoff_30 = forecast["date"].max() - timedelta(days=30)
            summary["risk_days_7"] = int(b[forecast["date"] >= cutoff_7].sum())
            summary["risk_days_30"] = int(b[forecast["date"] >= cutoff_30].sum())

        # Recurrence probability heuristic:
        # If your pipeline already computes next-7-day risk, prefer that column.
        if "recurrence_7d_prob" in forecast.columns:
            summary["recurrence_7d_prob"] = float(last_row["recurrence_7d_prob"]) if not pd.isna(last_row["recurrence_7d_prob"]) else None
        else:
            # Heuristic: blend today risk_prob with recent risk frequency
            rp = summary["risk_prob"] if summary["risk_prob"] is not None and not np.isnan(summary["risk_prob"]) else 0.0
            if b is not None:
                recent_freq = b.tail(7).mean() if len(b) >= 1 else 0.0
            else:
                recent_freq = 0.0
            # Weighted combination
            summary["recurrence_7d_prob"] = float(np.clip(0.6 * rp + 0.4 * recent_freq, 0, 1))

        summary["last_update"] = forecast["date"].max()

    # If env history has CHL, also extract latest CHL if missing
    if summary["latest_chl"] is None and not env.empty:
        for c in ["CHL", "CHL_MG_M3", "CHLOROPHYLL"]:
            if c in env.columns:
                summary["latest_chl"] = float(env[c].dropna().iloc[-1])
                break

    return summary


def plot_timeseries(df: pd.DataFrame, x: str, y: str, title: str, ylab: str):
    fig = px.line(
        df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title=ylab)
    fig.update_xaxes(title=None)
    return fig


def make_region_polygon(bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    coords = [
        [lat_min, lon_min],
        [lat_min, lon_max],
        [lat_max, lon_max],
        [lat_max, lon_min],
        [lat_min, lon_min],
    ]
    return coords


# =========================
# ---- SIDEBAR & HEADER
# =========================
with st.sidebar:
    st.markdown("### 🌊 MARS – Marine Autonomous Risk System")
    st.write("Part of **Annamaria Souri**’s PhD research • Powered by **Copernicus Marine**")
    data_dir = st.text_input("Data directory", value=str(DEFAULT_DATA_DIR), help="Folder containing the daily CSVs")
    st.markdown("---")
    st.caption("Tip: If the map click doesn't switch regions, use the selector above the map.")

st.title("MARS – Marine Autonomous Risk System")
sub = st.columns([1,1,1,2])
with sub[0]:
    st.markdown("**Today’s Forecast**")
with sub[3]:
    st.markdown("<div style='text-align:right;color:#555'>Updated automatically each day</div>", unsafe_allow_html=True)

# =========================
# ---- DATA LOADING (cached)
# =========================
@st.cache_data(ttl=3600)
def get_region_data(region_key: str, base_dir: str):
    base = Path(base_dir)
    forecast = load_forecast_log(base, region_key)
    env = load_env_history(base, region_key)
    summary = summarize_region(forecast, env)
    return forecast, env, summary


# =========================
# ---- MAP (clickable regions)
# =========================

# Derive a center for the map (midpoint of all bboxes)
all_lats = []
all_lons = []
for r in REGIONS.values():
    lat_min, lat_max, lon_min, lon_max = r["bbox"]
    all_lats += [lat_min, lat_max]
    all_lons += [lon_min, lon_max]
center_lat = float(np.mean(all_lats))
center_lon = float(np.mean(all_lons))

# Selector as fallback
region_keys = list(REGIONS.keys())
if "selected_region" not in st.session_state:
    st.session_state.selected_region = region_keys[0]

selector_cols = st.columns([3,1])
with selector_cols[0]:
    st.markdown("#### Regions Map (click a box to select)")
with selector_cols[1]:
    st.selectbox("Active region", options=region_keys, format_func=lambda k: REGIONS[k]["title"], key="selected_region")

fol_map = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="cartodbpositron")

for key, meta in REGIONS.items():
    coords = make_region_polygon(meta["bbox"])  # list of [lat, lon]
    # Folium expects [lat, lon]
    gj = folium.GeoJson(
        {
            "type": "Feature",
            "properties": {"region": key, "title": meta["title"]},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[ [c[1], c[0]] for c in coords ]],  # folium uses [lon, lat]
            },
        },
        style_function=lambda feat, color=meta["color"]: {
            "fillColor": color,
            "color": color,
            "weight": 2,
            "fillOpacity": 0.15 if key != st.session_state.selected_region else 0.35,
        },
        highlight_function=lambda x: {"weight": 3, "fillOpacity": 0.3},
        tooltip=folium.Tooltip(f"<b>{meta['title']}</b>", sticky=True),
        name=meta["title"],
    )

    gj.add_child(
        folium.features.GeoJsonPopup(fields=["region"], aliases=["Key:"])
    )

    def on_click_factory(region_key):
        # Folium click is captured at Python-level as a map click; we infer selection by bounds.
        # We can't attach per-feature click in st_folium reliably across versions, so we rely on bbox hit-testing below.
        return None

    gj.add_to(fol_map)

# Capture map click
mret = st_folium(fol_map, width=None, height=450, returned_objects=["last_object_clicked", "last_clicked"], key="mars_map")
if mret and mret.get("last_clicked"):
    clat = mret["last_clicked"]["lat"]
    clon = mret["last_clicked"]["lng"]
    # Hit-test against region bboxes
    for key, meta in REGIONS.items():
        lat_min, lat_max, lon_min, lon_max = meta["bbox"]
        if (lat_min <= clat <= lat_max) and (lon_min <= clon <= lon_max):
            st.session_state.selected_region = key
            break

# =========================
# ---- LOAD SELECTED REGION
# =========================
region_key = st.session_state.selected_region
forecast_df, env_df, summary = get_region_data(region_key, data_dir)
region_title = REGIONS[region_key]["title"]

# =========================
# ---- KPI CARDS
# =========================

kpi = st.columns(6)

fmt_chl = f"{summary['latest_chl']:.3f} mg/m³" if summary["latest_chl"] is not None else "—"
fmt_prob = f"{summary['risk_prob']*100:.0f}%" if summary["risk_prob"] is not None and not np.isnan(summary["risk_prob"]) else "—"
fmt_flag = "Yes" if summary["bloom_flag"] else ("No" if summary["bloom_flag"] is not None else "—")
fmt_thr = f"{summary['threshold']}" if summary["threshold"] is not None and not (isinstance(summary['threshold'], float) and np.isnan(summary['threshold'])) else "—"
fmt_7 = str(summary["risk_days_7"]) if summary["risk_days_7"] is not None else "—"
fmt_30 = str(summary["risk_days_30"]) if summary["risk_days_30"] is not None else "—"

with kpi[0]:
    st.metric(label=f"{region_title}: Latest CHL", value=fmt_chl)
with kpi[1]:
    st.metric(label="Bloom Flag (Today)", value=fmt_flag)
with kpi[2]:
    st.metric(label="Threshold Used", value=fmt_thr)
with kpi[3]:
    st.metric(label="Risk Probability (Today)", value=fmt_prob)
with kpi[4]:
    st.metric(label="Risk Days (7d)", value=fmt_7)
with kpi[5]:
    st.metric(label="Risk Days (30d)", value=fmt_30)

# Recurrence banner
rec_prob = summary["recurrence_7d_prob"]
if rec_prob is not None:
    st.markdown(
        f"""
        <div style='background:#f4f6f8;border:1px solid #e0e0e0;padding:12px 16px;border-radius:12px;margin:8px 0;'>
            <b>Risk likelihood:</b> There is a <b>{rec_prob*100:.0f}%</b> chance of bloom recurrence in the next 7 days.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# ---- TABS
# =========================

TAB_LABELS = ["Today’s Forecast", "Environmental Trends", "About the MARS System"]

if "CHL" not in [c.upper() for c in env_df.columns] and not forecast_df.empty:
    # Try to infer CHL into env_df from forecast timeline to enable time-series plots
    if "date" in forecast_df.columns:
        env_df = env_df.copy()
        env_df["DATETIME"] = pd.to_datetime(env_df["DATETIME"]) if "DATETIME" in env_df.columns else pd.to_datetime([])


 t1, t2, t3 = st.tabs(TAB_LABELS)

# ---- Tab 1: Today’s Forecast
with t1:
    st.subheader(f"{region_title} – Today’s Forecast")
    if forecast_df.empty:
        st.info("No forecast_log file found for this region yet. Check your data directory.")
    else:
        # 7-day and 30-day CHL time-series
        # Use env_df for CHL if present, else forecast_df
        chl_source = None
        chl_col = None
        if not env_df.empty:
            for c in ["CHL", "CHL_MG_M3", "CHLOROPHYLL"]:
                if c in env_df.columns:
                    chl_source = env_df.rename(columns={c: "CHL"})[["DATETIME", "CHL"]].dropna()
                    chl_source = chl_source.sort_values("DATETIME")
                    break
        if chl_source is None and "chl_pred" in forecast_df.columns:
            tmp = forecast_df[["date", "chl_pred"]].rename(columns={"date": "DATETIME", "chl_pred": "CHL"}).dropna()
            chl_source = tmp.sort_values("DATETIME")
        elif chl_source is None and "chl" in forecast_df.columns:
            tmp = forecast_df[["date", "chl"]].rename(columns={"date": "DATETIME", "chl": "CHL"}).dropna()
            chl_source = tmp.sort_values("DATETIME")

        if chl_source is None or chl_source.empty:
            st.warning("CHL series not available to plot.")
        else:
            now = chl_source["DATETIME"].max()
            last7 = chl_source[chl_source["DATETIME"] >= now - timedelta(days=7)]
            last30 = chl_source[chl_source["DATETIME"] >= now - timedelta(days=30)]

            c1, c2 = st.columns(2)
            with c1:
                fig7 = plot_timeseries(last7, x="DATETIME", y="CHL", title="CHL – Last 7 days", ylab="mg/m³")
                st.plotly_chart(fig7, use_container_width=True)
            with c2:
                fig30 = plot_timeseries(last30, x="DATETIME", y="CHL", title="CHL – Last 30 days", ylab="mg/m³")
                st.plotly_chart(fig30, use_container_width=True)

        # Threshold & probability distribution (if available)
        if "risk_prob" in forecast_df.columns:
            figp = px.line(
                forecast_df.tail(30), x="date", y="risk_prob", title="Risk Probability (last 30 days)", template=PLOTLY_TEMPLATE
            )
            figp.update_yaxes(range=[0,1], tickformat=",.0%", title="Probability")
            figp.update_xaxes(title=None)
            st.plotly_chart(figp, use_container_width=True)

# ---- Tab 2: Environmental Trends
with t2:
    st.subheader(f"{region_title} – Environmental Trends (30 days)")
    if env_df.empty:
        st.info("No env_history file found for this region yet.")
    else:
        # Variable picker
        var_labels = {k:v for k,v in ENV_VARS}
        available = [k for k,_ in ENV_VARS if k in env_df.columns]
        if not available:
            st.warning("No known environmental variable columns found.")
        else:
            picked = st.multiselect("Variables", options=available, default=[available[0]])
            for var in picked:
                pretty = dict(ENV_VARS).get(var, var)
                dfv = env_df[["DATETIME", var]].dropna()
                if dfv.empty:
                    continue
                figv = plot_timeseries(dfv, x="DATETIME", y=var, title=pretty, ylab=pretty)
                st.plotly_chart(figv, use_container_width=True)

# ---- Tab 3: About
with t3:
    st.subheader("About the MARS System")
    st.markdown(
        """
        **MARS (Marine Autonomous Risk System)** is a research dashboard forecasting red tide (harmful algal bloom) risk
        in the Eastern Mediterranean using daily **Copernicus Marine** data and a trained machine learning model.

        **Regions covered:** Thermaikos (Greece), Piraeus (Greece), and Limassol (Cyprus).

        This dashboard displays:
        - A full-width interactive region map (click to switch)
        - Today’s CHL estimate, bloom flag, threshold, and recent risk-day counts
        - CHL time-series (7 & 30 days)
        - Environmental trends for NH₄, NO₃, PO₄, temperature (θ / SST), and salinity (SAL)

        **Credits:** Part of *Annamaria Souri*’s PhD research • Powered by *Copernicus Marine*.
        """
    )

# =========================
# ---- FOOTER / NOTES
# =========================

st.markdown(
    """
    <div style='text-align:center;color:#999;font-size:12px;margin-top:24px;'>
    © {year} MARS • Research prototype – not for operational navigation.
    </div>
    """.format(year=datetime.now().year),
    unsafe_allow_html=True,
)

