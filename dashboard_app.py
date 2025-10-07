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

# --- Marine Observatory theme ---
PRIMARY_DARK = "#062B4F"      # deep ocean
PRIMARY = "#0B4F6C"           # marine blue
PRIMARY_GRAD_1 = "#0072BC"    # Copernicus blue
PRIMARY_GRAD_2 = "#00B4D8"    # aqua
ACCENT = "#34D1BF"            # teal accent
AMBER = "#FFB703"
RED = "#D00000"
GREEN = "#2A9D8F"
MUTED = "#6B7A90"
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

# === THEME CSS ===
st.markdown(
    f"""
    <style>
      :root {{
        --grad1: {PRIMARY_GRAD_1};
        --grad2: {PRIMARY_GRAD_2};
        --primary: {PRIMARY};
        --dark: {PRIMARY_DARK};
        --muted: {MUTED};
        --green: {GREEN};
        --amber: {AMBER};
        --red: {RED};
        --accent: {ACCENT};
      }}
      .marine-hero {{
        background: linear-gradient(90deg, var(--grad1), var(--grad2));
        color: white;
        padding: 18px 22px; border-radius: 16px; box-shadow: 0 10px 28px rgba(0,0,0,.08);
        position: relative; overflow: hidden;
      }}
      .wave {{
        position:absolute;bottom:-10px;left:-5%;right:-5%;height:80px;
        background: radial-gradient(ellipse at bottom, rgba(255,255,255,.35), rgba(255,255,255,0));
        filter: blur(14px);
        animation: swell 6s ease-in-out infinite;
      }}
      @keyframes swell {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(6px); }}
        100% {{ transform: translateY(0px); }}
      }}
      .kpi {{
        background: #ffffff; border: 1px solid rgba(0,0,0,.06); border-radius: 14px;
        padding: 14px 16px; box-shadow: 0 6px 20px rgba(13, 51, 89, .06);
      }}
      .kpi .label {{ color: var(--muted); font-size: 13px; letter-spacing: .2px; }}
      .kpi .value {{ font-size: 22px; font-weight: 700; color: var(--dark); }}
      .badge {{ display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:12px;}}
      .badge.low {{ background: rgba(42,157,143,.12); color: var(--green); border:1px solid rgba(42,157,143,.4); }}
      .badge.med {{ background: rgba(255,183,3,.12); color: var(--amber); border:1px solid rgba(255,183,3,.4); }}
      .badge.high {{ background: rgba(208,0,0,.10); color: var(--red); border:1px solid rgba(208,0,0,.35); }}
      .section-title {{ color: var(--dark); font-weight:800; }}
      .soft-card {{ background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:16px;padding:14px;box-shadow:0 8px 24px rgba(0,0,0,.05);}}
    </style>
    """,
    unsafe_allow_html=True,
)

# === SIDEBAR ===
with st.sidebar:
    st.markdown("### 🌊 MARS – Marine Autonomous Risk System")
    st.write("Part of **Annamaria Souri**’s PhD research • Powered by **Copernicus Marine**")

# ✅ Use current working directory for data files
data_dir = "."

# --- Hero Header ---
st.markdown(
    """
    <div class="marine-hero">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:28px;">🛰️</div>
        <div>
          <div style="font-size:22px;font-weight:800;letter-spacing:.3px;">MARS Dashboard</div>
          <div style="opacity:.9">Real‑Time Bloom Forecasts for the Eastern Mediterranean</div>
        </div>
        <div style="margin-left:auto;opacity:.9;">Updated daily</div>
      </div>
      <div class="wave"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# === HELPERS ===

def _list_files():
    try:
        return sorted(os.listdir(data_dir))
    except Exception:
        return []


def latest_env_file(region: str):
    files = [f for f in _list_files() if re.match(rf"env_history_{region}_.+\\.csv$", f, flags=re.IGNORECASE)]
    return os.path.join(data_dir, sorted(files)[-1]) if files else None


def load_forecast(region: str) -> pd.DataFrame:
    for name in [f"forecast_log_{region}.csv", f"forecast_{region}.csv"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            for c in ["date", "day", "ds", "timestamp"]:
                if c in df.columns:
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break
            df = df.rename(columns={
                "bloom_flag":"bloom_risk_flag",
                "risk_flag":"bloom_risk_flag",
                "chl_pred":"predicted_chl",
                "threshold":"threshold_used",
            })
            return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame()


def load_env(region: str) -> pd.DataFrame:
    f = latest_env_file(region)
    if not f:
        return pd.DataFrame()
    df = pd.read_csv(f)
    df.columns = [c.strip().upper() for c in df.columns]
    for alias in ["DATETIME", "DATE", "TS"]:
        if alias in df.columns and "TIME" not in df.columns:
            df = df.rename(columns={alias: "TIME"})
    return df


def plot_ts(df: pd.DataFrame, x: str, y: str, title: str, ylab: str):
    fig = px.line(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE,
                  color_discrete_sequence=[PRIMARY_GRAD_1])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title=ylab)
    return fig


def summarize_region(forecast: pd.DataFrame) -> dict:
    out = {"latest_chl": None, "bloom_flag": None, "threshold": None,
           "rec7": None, "rec30": None, "risk7": None, "risk30": None}
    if forecast.empty:
        return out
    last = forecast.dropna(subset=["date"]).iloc[-1] if "date" in forecast.columns else forecast.iloc[-1]
    out["latest_chl"] = last.get("predicted_chl")
    out["threshold"] = last.get("threshold_used")
    bf = last.get("bloom_risk_flag")
    out["bloom_flag"] = (str(bf).lower() in ("1","true","yes")) if pd.notna(bf) else None

    if "bloom_risk_flag" in forecast.columns:
        flags = forecast["bloom_risk_flag"].astype(str).str.lower().isin(["1","true","yes"]) 
    elif {"predicted_chl","threshold_used"}.issubset(forecast.columns):
        flags = forecast["predicted_chl"] >= forecast["threshold_used"]
    else:
        flags = pd.Series([False]*len(forecast), index=forecast.index)

    if "date" in forecast.columns:
        fc = forecast.dropna(subset=["date"]).copy(); fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
        fc = fc.dropna(subset=["date"]) 
        end = fc["date"].max()
        w7 = fc[fc["date"] >= end - timedelta(days=7)].index
        w30 = fc[fc["date"] >= end - timedelta(days=30)].index
        out["risk7"] = int(flags.loc[w7].sum()) if len(w7) else 0
        out["risk30"] = int(flags.loc[w30].sum()) if len(w30) else 0
        out["rec7"] = float(last.get("recurrence_7d_prob")) if "recurrence_7d_prob" in forecast.columns and pd.notna(last.get("recurrence_7d_prob")) else (round(flags.loc[w7].mean()*100,1) if len(w7) else None)
        out["rec30"] = float(last.get("recurrence_30d_prob")) if "recurrence_30d_prob" in forecast.columns and pd.notna(last.get("recurrence_30d_prob")) else (round(flags.loc[w30].mean()*100,1) if len(w30) else None)
    else:
        out["risk7"], out["risk30"] = int(flags.tail(7).sum()), int(flags.tail(30).sum())
        out["rec7"], out["rec30"] = round(flags.tail(7).mean()*100,1), round(flags.tail(30).mean()*100,1)
    return out

# === MAP ===
all_lat, all_lon = [], []
for v in REGIONS.values():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    all_lat += [lat_min, lat_max]; all_lon += [lon_min, lon_max]
center = [float(np.mean(all_lat)), float(np.mean(all_lon))]

available = [r for r in REGIONS if os.path.exists(os.path.join(data_dir, f"forecast_log_{r}.csv"))]
if "region" not in st.session_state:
    st.session_state.region = available[0] if available else "thermaikos"

head_cols = st.columns([3,1])
with head_cols[0]:
    st.markdown("<div class='section-title' style='margin:16px 0 6px;'>📍 Regions Map (click to select)</div>", unsafe_allow_html=True)
with head_cols[1]:
    st.selectbox("Active region", options=list(REGIONS.keys()), format_func=lambda k: REGIONS[k]["title"], key="region")

m = folium.Map(location=center, zoom_start=7, tiles="cartodbpositron")
for k, v in REGIONS.items():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    folium.Rectangle(bounds=[[lat_min, lon_min],[lat_max, lon_max]], color=v["color"], fill=True,
                     fill_opacity=0.25 if k != st.session_state.region else 0.5, popup=v["title"]).add_to(m)

st_folium(m, height=420, key="mars_map")

region = st.session_state.region
forecast = load_forecast(region)
env = load_env(region)
region_title = REGIONS[region]["title"]
summary = summarize_region(forecast)

# === KPI CARDS ===

def likelihood_badge(pct: float | None) -> str:
    if pct is None:
        return "<span class='badge'>—</span>"
    if pct <= 20: cls, label = "low", "Low"
    elif pct <= 60: cls, label = "med", "Moderate"
    else: cls, label = "high", "High"
    return f"<span class='badge {cls}'>{pct:.0f}% • {label}</span>"

k1, k2, k3 = st.columns([2,2,3])
with k1:
    st.markdown(f"""
    <div class='kpi'>
      <div class='label'>{region_title} – CHL</div>
      <div class='value'>{summary['latest_chl']:.3f} mg/m³</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class='kpi'>
      <div class='label'>Bloom Flag (Today)</div>
      <div class='value'>{'Yes' if summary['bloom_flag'] else ('No' if summary['bloom_flag'] is not None else '—')}</div>
    </div>""", unsafe_allow_html=True)
with k3:
 st.markdown(f"""
    <div class='kpi'>
      <div class='label'>Threshold Used</div>
      <div class='value'>
        {f"{summary['threshold']:.3f}" if isinstance(summary['threshold'], (int, float)) and not pd.isna(summary['threshold']) else "—"}
      </div>
    </div>
""", unsafe_allow_html=True)


k4, k5, k6 = st.columns([3,3,2])
with k4:
    st.markdown(f"<div class='kpi'><div class='label'>Likelihood (Next 7 d)</div>{likelihood_badge(summary['rec7'])}</div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi'><div class='label'>Likelihood (Next 30 d)</div>{likelihood_badge(summary['rec30'])}</div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='kpi'><div class='label'>Risk Days (7/30)</div><div class='value'>{summary['risk7'] or 0}/{summary['risk30'] or 0}</div></div>", unsafe_allow_html=True)

# === TABS ===
tab1, tab2, tab3 = st.tabs(["Today’s Forecast", "Environmental Trends", "About MARS"])

with tab1:
    st.markdown("<div class='section-title'>CHL Forecasts</div>", unsafe_allow_html=True)
    if not env.empty and "TIME" in env.columns and "CHL" in env.columns:
        env = env.copy(); env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce"); env = env.dropna(subset=["TIME"])
        now = env["TIME"].max()
        last7 = env[env["TIME"] >= now - timedelta(days=7)]
        last30 = env[env["TIME"] >= now - timedelta(days=30)]
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_ts(last7, "TIME", "CHL", "CHL – Last 7 days", "mg/m³"), use_container_width=True)
        with c2: st.plotly_chart(plot_ts(last30, "TIME", "CHL", "CHL – Last 30 days", "mg/m³"), use_container_width=True)
    elif not forecast.empty and {"date","predicted_chl"}.issubset(forecast.columns):
        st.info("Using predicted CHL from forecast history (env history not found).")
        st.plotly_chart(px.line(forecast.tail(30), x="date", y="predicted_chl", title="Predicted CHL (last 30 days)",
                                color_discrete_sequence=[PRIMARY_GRAD_2], template=PLOTLY_TEMPLATE), use_container_width=True)
    else:
        st.info("No environmental or forecast CHL series available yet.")

with tab2:
    st.markdown(f"<div class='section-title'>{region_title} – Environmental Trends (30 days)</div>", unsafe_allow_html=True)
    if env.empty or "TIME" not in env.columns:
        st.info("No env_history file with a TIME column found for this region yet.")
    else:
        env = env.copy(); env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce")
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
        <div class='soft-card'>
        <h3 style='margin:0 0 10px 0;'>About MARS</h3>
        <p><b>MARS – Marine Autonomous Risk System</b> forecasts harmful algal bloom (red tide) risk in the
        Eastern Mediterranean using daily <b>Copernicus Marine</b> data and a trained machine‑learning model.</p>
        <p><b>Regions:</b> Thermaikos (GR), Piraeus (GR), Limassol (CY).<br/>
           <b>Variables:</b> NH₄, NO₃, PO₄, θ (temperature), SO (salinity), CHL.</p>
        <p><i>Part of Annamaria Souri’s PhD Research – University of Nicosia.</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Diagnostics ---
with st.expander("🔍 Diagnostics"):
    st.write("Working directory:", os.getcwd())
    st.write("Files:", _list_files())
    st.write("Forecast columns:", list(forecast.columns))
    st.write("Env columns:", list(env.columns))

st.markdown(
    f"<hr style='margin-top:2em;border:0;height:2px;background:linear-gradient(90deg,{PRIMARY_GRAD_1},{PRIMARY_GRAD_2});opacity:.8;'>"
    f"<div style='text-align:center;color:{MUTED};font-size:12px;'>© {datetime.now().year} MARS • Research prototype</div>",
    unsafe_allow_html=True,
)

