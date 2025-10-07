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
    st.markdown("### 🌊 MARS – Marine Autonomous Risk System")
    st.write("Part of **Annamaria Souri**’s PhD research • Powered by **Copernicus Marine**")

# ✅ Use current working directory for data files
data_dir = "."

st.title("🌊 MARS Dashboard – Real-Time Bloom Forecasts")

# === HELPERS ===

def _list_files():
    try:
        return sorted(os.listdir(data_dir))
    except Exception:
        return []


def latest_env_file(region: str):
    # Accept both hyphen and underscore dates, any suffix
    files = [f for f in _list_files() if re.match(rf"env_history_{region}_.+\.csv$", f, flags=re.IGNORECASE)]
    return os.path.join(data_dir, sorted(files)[-1]) if files else None


def load_forecast(region: str) -> pd.DataFrame:
    # Support different casings
    candidates = [f"forecast_log_{region}.csv", f"forecast_{region}.csv"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            # Parse date
            for c in ["date", "day", "ds", "timestamp"]:
                if c in df.columns:
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break
            # Normalize flag/threshold names
            rename = {
                "bloom_flag": "bloom_risk_flag",
                "risk_flag": "bloom_risk_flag",
                "chl_pred": "predicted_chl",
                "threshold": "threshold_used",
            }
            for k, v in rename.items():
                if k in df.columns and v not in df.columns:
                    df = df.rename(columns={k: v})
            return df.sort_values("date")
    return pd.DataFrame()


def load_env(region: str) -> pd.DataFrame:
    f = latest_env_file(region)
    if not f:
        return pd.DataFrame()
    df = pd.read_csv(f)
    # normalize columns
    cols_upper = {c: c.upper() for c in df.columns}
    df = df.rename(columns=cols_upper)
    # allow TIME aliases
    for alias in ["DATETIME", "DATE", "TS"]:
        if alias in df.columns and "TIME" not in df.columns:
            df = df.rename(columns={alias: "TIME"})
    return df


def plot_ts(df: pd.DataFrame, x: str, y: str, title: str, ylab: str):
    fig = px.line(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[COPERNICUS_BLUE])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title=ylab)
    return fig


def summarize_region(forecast: pd.DataFrame) -> dict:
    """Return latest metrics + derived 7/30 day likelihoods and risk-day counts.
    If recurrence columns exist, use them; otherwise compute from predicted_chl vs threshold_used.
    """
    out = {
        "latest_chl": None,
        "bloom_flag": None,
        "threshold": None,
        "rec7": None,
        "rec30": None,
        "risk7": None,
        "risk30": None,
    }
    if forecast.empty:
        return out

    last = forecast.dropna(subset=["date"]).iloc[-1] if "date" in forecast.columns else forecast.iloc[-1]

    # Latest values
    out["latest_chl"] = last.get("predicted_chl")
    out["threshold"] = last.get("threshold_used")
    bf = last.get("bloom_risk_flag")
    if pd.isna(bf):
        out["bloom_flag"] = None
    else:
        out["bloom_flag"] = str(bf).lower() in ("1", "true", "yes")

    # Compute risk flags series
    if "bloom_risk_flag" in forecast.columns:
        flags = forecast["bloom_risk_flag"].astype(str).str.lower().isin(["1", "true", "yes"])
    elif {"predicted_chl", "threshold_used"}.issubset(forecast.columns):
        flags = forecast["predicted_chl"] >= forecast["threshold_used"]
    else:
        flags = pd.Series([False] * len(forecast), index=forecast.index)

    # Windows
    if "date" in forecast.columns:
        fc = forecast.dropna(subset=["date"]).copy()
        fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
        fc = fc.dropna(subset=["date"]).reset_index(drop=True)
        end = fc["date"].max()
        w7 = fc[fc["date"] >= end - timedelta(days=7)].index
        w30 = fc[fc["date"] >= end - timedelta(days=30)].index
        # counts
        out["risk7"] = int(flags.loc[w7].sum()) if len(w7) else 0
        out["risk30"] = int(flags.loc[w30].sum()) if len(w30) else 0
        # recurrence probabilities
        if "recurrence_7d_prob" in forecast.columns and pd.notna(last.get("recurrence_7d_prob")):
            out["rec7"] = float(last.get("recurrence_7d_prob"))
        else:
            out["rec7"] = round(flags.loc[w7].mean() * 100, 1) if len(w7) else None
        if "recurrence_30d_prob" in forecast.columns and pd.notna(last.get("recurrence_30d_prob")):
            out["rec30"] = float(last.get("recurrence_30d_prob"))
        else:
            out["rec30"] = round(flags.loc[w30].mean() * 100, 1) if len(w30) else None
    else:
        # fallback when no dates exist
        out["risk7"] = int(flags.tail(7).sum())
        out["risk30"] = int(flags.tail(30).sum())
        out["rec7"] = round(flags.tail(7).mean() * 100, 1)
        out["rec30"] = round(flags.tail(30).mean() * 100, 1)

    return out

# === MAP ===
all_lat, all_lon = [], []
for v in REGIONS.values():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    all_lat += [lat_min, lat_max]
    all_lon += [lon_min, lon_max]
center = [float(np.mean(all_lat)), float(np.mean(all_lon))]

# Choose default region that actually has data
available = [r for r in REGIONS if os.path.exists(os.path.join(data_dir, f"forecast_log_{r}.csv"))]
if "region" not in st.session_state:
    st.session_state.region = available[0] if available else "thermaikos"

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
summary = summarize_region(forecast)

# === KPI CARDS ===
kpi = st.columns(6)

kpi[0].metric(f"{region_title} – CHL", f"{summary['latest_chl']:.3f} mg/m³" if pd.notna(summary['latest_chl']) else "—")
kpi[1].metric("Bloom Flag (Today)", "Yes" if summary['bloom_flag'] else ("No" if summary['bloom_flag'] is not None else "—"))
kpi[2].metric("Threshold Used", f"{summary['threshold']:.3f}" if pd.notna(summary['threshold']) else "—")
kpi[3].metric("Likelihood (Next 7 d)", f"{summary['rec7']} %" if summary['rec7'] is not None else "—")
kpi[4].metric("Likelihood (Next 30 d)", f"{summary['rec30']} %" if summary['rec30'] is not None else "—")
kpi[5].metric("Risk Days (7/30)", f"{summary['risk7'] or 0}/{summary['risk30'] or 0}")

# === TABS ===
tab1, tab2, tab3 = st.tabs(["Today’s Forecast", "Environmental Trends", "About MARS"])

with tab1:
    st.subheader(f"{region_title} – CHL Forecasts")
    # CHL series from env if available; else predicted CHL from forecast
    if not env.empty and "TIME" in env.columns and "CHL" in env.columns:
        env = env.copy()
        env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce")
        env = env.dropna(subset=["TIME"])  # keep valid timestamps
        now = env["TIME"].max()
        last7 = env[env["TIME"] >= now - timedelta(days=7)]
        last30 = env[env["TIME"] >= now - timedelta(days=30)]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_ts(last7, "TIME", "CHL", "CHL – Last 7 days", "mg/m³"), use_container_width=True)
        with c2:
            st.plotly_chart(plot_ts(last30, "TIME", "CHL", "CHL – Last 30 days", "mg/m³"), use_container_width=True)
    elif not forecast.empty and {"date", "predicted_chl"}.issubset(forecast.columns):
        st.info("Using predicted CHL from forecast history (env history not found).")
        st.plotly_chart(
            px.line(
                forecast.tail(30), x="date", y="predicted_chl",
                title="Predicted CHL (last 30 days)",
                color_discrete_sequence=[COPERNICUS_AQUA],
                template=PLOTLY_TEMPLATE,
            ),
            use_container_width=True,
        )
    else:
        st.info("No environmental or forecast CHL series available yet.")

with tab2:
    st.subheader(f"{region_title} – Environmental Trends (30 days)")
    if env.empty or "TIME" not in env.columns:
        st.info("No env_history file with a TIME column found for this region yet.")
    else:
        env = env.copy()
        env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce")
        variables = [v for v, _ in ENV_VARS if v in env.columns]
        if not variables:
            st.info("No known environmental variables present.")
        else:
            chosen = st.multiselect("Variables to plot", variables, default=variables[:2])
            for v in chosen:
                label = dict(ENV_VARS).get(v, v)
                st.plotly_chart(plot_ts(env, "TIME", v, label, label), use_container_width=True)

with tab3:
    st.markdown(
        """
        **MARS – Marine Autonomous Risk System** forecasts harmful algal bloom (red tide) risk in the
        Eastern Mediterranean using daily **Copernicus Marine** data and a trained machine-learning model.

        **Regions:** Thermaikos (GR), Piraeus (GR), Limassol (CY)  
        **Data:** NH₄, NO₃, PO₄, θ (temperature), SO (salinity), CHL  
        **Metrics shown:** Bloom flag, thresholds, risk probabilities, and 7/30-day recurrence likelihoods.  

        *Part of Annamaria Souri’s PhD Research – University of Nicosia*
        """
    )

# --- Optional diagnostics (collapsed) ---
with st.expander("🔍 Diagnostics (what the app finds)"):
    st.write("Working directory:", os.getcwd())
    st.write("Files:", _list_files())
    st.write("Forecast columns:", list(forecast.columns))
    st.write("Env columns:", list(env.columns))

st.markdown(
    f"<hr style='margin-top:2em;border:0;height:1px;background:{COPERNICUS_BLUE};opacity:0.3;'>"
    f"<div style='text-align:center;color:#999;font-size:12px;'>© {datetime.now().year} MARS • Research prototype</div>",
    unsafe_allow_html=True,
)
