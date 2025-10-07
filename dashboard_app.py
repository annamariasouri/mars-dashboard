import os, re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

# === CONFIG ===
st.set_page_config(page_title="MARS – Marine Autonomous Risk System", page_icon="🌊", layout="wide")

# --- Colors / Theme ---
PRIMARY_DARK = "#021F3B"
PRIMARY_GRAD_1 = "#004E89"
PRIMARY_GRAD_2 = "#00A6FB"
ACCENT = "#34D1BF"
AMBER = "#FFB703"
RED = "#E63946"
GREEN = "#2A9D8F"
MUTED = "#AAB4BF"
PLOTLY_TEMPLATE = "plotly_white"

# === Global CSS ===
st.markdown(f"""
<style>
/* ==== MARS Deep Ocean Theme ==== */
body {{
  background: linear-gradient(180deg, #00111f 0%, #001b33 40%, #001f3f 100%) fixed;
  color: #E0F2FF;
  font-family: 'Inter', sans-serif;
}}
.block-container {{
  background: transparent !important;
  padding-top: 0rem !important;
}}
.marine-hero {{
  background: linear-gradient(90deg, {PRIMARY_GRAD_1}, {PRIMARY_GRAD_2});
  color: white; border-radius: 22px;
  padding: 26px 30px; box-shadow: 0 10px 30px rgba(0,0,0,.6);
}}
.kpi {{
  background: #FFFFFF; /* White cards */
  border-radius: 18px;
  padding: 16px 20px;
  box-shadow: 0 8px 25px rgba(0,0,0,.4);
  text-align: center;
}}
.kpi .label {{
  color: #0B3954;       /* Deep marine blue text for labels */
  font-size: 14px;
  font-weight: 600;
  letter-spacing: .4px;
}}
.kpi .value {{
  font-size: 26px;
  font-weight: 800;
  color: #000000;       /* Pure black text for values */
}}
.badge {{
  display:inline-block;
  padding:6px 12px;
  border-radius:18px;
  font-size:14px;
  font-weight:700;
  line-height:1.2;
}}
.badge.low  {{ background: rgba(42,157,143,.15);  color:#005f4b;  border:1px solid rgba(42,157,143,.35);  }}
.badge.med  {{ background: rgba(255,183,3,.15);  color:#7c5700;  border:1px solid rgba(255,183,3,.40);  }}
.badge.high {{ background: rgba(230,57,70,.15);  color:#750000;  border:1px solid rgba(230,57,70,.35);  }}
.soft-card {{
  background: rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.1);
  border-radius:20px;
  padding:18px;
  box-shadow:0 8px 28px rgba(0,0,0,.5);
}}
/* === Map styling === */
iframe, .folium-map {{
  border-radius: 16px;
  box-shadow: 0 0 40px rgba(0,0,0,0.7);
}}
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.markdown("### 🌊 MARS – Marine Observatory Dashboard")
    st.write("**Annamaria Souri**, PhD Research • Powered by Copernicus Marine")

data_dir = "."

# === Header ===
st.markdown(f"""
<div class="marine-hero">
  <h2 style="margin:0;">🌊 MARS Dashboard</h2>
  <p style="margin:4px 0 0;">Real-Time Bloom Forecasts for the Eastern Mediterranean</p>
  <p style="margin:0;opacity:.9;">Updated daily</p>
</div>
""", unsafe_allow_html=True)

# === Regions ===
REGIONS = {
    "thermaikos": {"title": "Thermaikos (Greece)", "bbox": (40.2, 40.7, 22.5, 23.0), "color": "#0077B6"},
    "peiraeus":  {"title": "Piraeus (Greece)",   "bbox": (37.9, 38.1, 23.5, 23.8), "color": "#FF6B6B"},
    "limassol":  {"title": "Limassol (Cyprus)",  "bbox": (34.6, 34.8, 33.0, 33.2), "color": "#00B894"},
}

# === Helper functions ===
def _list_files():
    try:
        return sorted(os.listdir(data_dir))
    except Exception:
        return []

def latest_env_file(region):
    files = [f for f in _list_files() if re.match(rf"env_history_{region}_.+\\.csv$", f, flags=re.IGNORECASE)]
    return os.path.join(data_dir, sorted(files)[-1]) if files else None

def load_forecast(region):
    for name in [f"forecast_log_{region}.csv", f"forecast_{region}.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            # Parse common date column
            for c in ["date", "day", "ds", "timestamp"]:
                if c in df.columns:
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break
            return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame()

def load_env(region):
    f = latest_env_file(region)
    if not f:
        return pd.DataFrame()
    df = pd.read_csv(f)
    df.columns = [c.strip().upper() for c in df.columns]
    # auto-fix lowercase or DATE columns
    if "TIME" not in df.columns:
        for alias in ["TIME", "DATE", "DATETIME", "TS", "time", "date", "datetime", "ts"]:
            if alias.upper() in df.columns:
                df = df.rename(columns={alias.upper(): "TIME"})
                break
            if alias in df.columns:
                df = df.rename(columns={alias: "TIME"})
                break
    return df

def likelihood_badge(p):
    if p is None or pd.isna(p):
        return "<span class='badge'>—</span>"
    if p <= 20:
        return f"<span class='badge low'>{p:.0f}% • Low</span>"
    elif p <= 60:
        return f"<span class='badge med'>{p:.0f}% • Moderate</span>"
    return f"<span class='badge high'>{p:.0f}% • High</span>"

def compute_likelihoods(forecast: pd.DataFrame):
    """Return (rec7, rec30, risk7, risk30) based on history."""
    if forecast.empty:
        return None, None, 0, 0

    # If explicit columns exist, prefer them
    last = forecast.iloc[-1]
    rec7 = last.get("recurrence_7d_prob") if "recurrence_7d_prob" in forecast.columns else None
    rec30 = last.get("recurrence_30d_prob") if "recurrence_30d_prob" in forecast.columns else None

    # Compute flags
    if {"predicted_chl", "threshold_used"}.issubset(forecast.columns):
        flags = forecast["predicted_chl"] >= forecast["threshold_used"]
    elif "bloom_risk_flag" in forecast.columns:
        flags = forecast["bloom_risk_flag"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        return rec7, rec30, 0, 0

    # Windowed stats by date if available
    if "date" in forecast.columns:
        fc = forecast.dropna(subset=["date"]).copy()
        if fc.empty:
            return rec7, rec30, 0, 0
        fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
        fc = fc.dropna(subset=["date"]).reset_index(drop=True)
        end = fc["date"].max()
        idx7 = fc.index[fc["date"] >= end - timedelta(days=7)]
        idx30 = fc.index[fc["date"] >= end - timedelta(days=30)]
        risk7 = int(flags.loc[idx7].sum()) if len(idx7) else 0
        risk30 = int(flags.loc[idx30].sum()) if len(idx30) else 0
        if rec7 is None and len(idx7):
            rec7 = round(flags.loc[idx7].mean() * 100, 1)
        if rec30 is None and len(idx30):
            rec30 = round(flags.loc[idx30].mean() * 100, 1)
        return rec7, rec30, risk7, risk30

    # Fallback when no dates
    risk7 = int(flags.tail(7).sum())
    risk30 = int(flags.tail(30).sum())
    if rec7 is None:
        rec7 = round(flags.tail(7).mean() * 100, 1)
    if rec30 is None:
        rec30 = round(flags.tail(30).mean() * 100, 1)
    return rec7, rec30, risk7, risk30

# === Map ===
region = st.session_state.get("region", "thermaikos")
lat_center = np.mean([REGIONS[r]["bbox"][0] + REGIONS[r]["bbox"][1] for r in REGIONS]) / 2
lon_center = np.mean([REGIONS[r]["bbox"][2] + REGIONS[r]["bbox"][3] for r in REGIONS]) / 2

m = folium.Map(location=[lat_center, lon_center], zoom_start=6, tiles="cartodbpositron")
for key, val in REGIONS.items():
    lat_min, lat_max, lon_min, lon_max = val["bbox"]
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color=val["color"], fill=True,
        fill_opacity=0.4, popup=val["title"]
    ).add_to(m)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#A9D6E5;font-weight:600;'>📍 Click a region below</div>", unsafe_allow_html=True)
click = st_folium(m, height=600, width=1400, key="mars_map")  # Bigger map

if click and click.get("last_clicked"):
    lat, lon = click["last_clicked"]["lat"], click["last_clicked"]["lng"]
    for k, v in REGIONS.items():
        lat_min, lat_max, lon_min, lon_max = v["bbox"]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            region = k
st.session_state.region = region
region_title = REGIONS[region]["title"]

# === Data load ===
forecast = load_forecast(region)
env = load_env(region)

# === KPI Section ===
st.markdown(f"<h3 style='color:#A9D6E5;margin-top:1em;'>{region_title}</h3>", unsafe_allow_html=True)
cols_top = st.columns(3)

with cols_top[0]:
    val = forecast["predicted_chl"].iloc[-1] if not forecast.empty and "predicted_chl" in forecast.columns else None
    val_display = f"{val:.3f}" if isinstance(val, (int, float)) and not pd.isna(val) else "—"
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>Predicted CHL</div>
          <div class='value'>{val_display} mg/m³</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols_top[1]:
    thr = forecast["threshold_used"].iloc[-1] if "threshold_used" in forecast.columns and not forecast.empty else None
    thr_display = f"{thr:.3f}" if isinstance(thr, (int, float)) and not pd.isna(thr) else "—"
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>Threshold</div>
          <div class='value'>{thr_display}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols_top[2]:
    rf = forecast["bloom_risk_flag"].iloc[-1] if "bloom_risk_flag" in forecast.columns and not forecast.empty else None
    rf_label = "Yes" if str(rf).lower() in ("1", "true", "yes") else ("No" if rf is not None else "—")
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>Bloom Today</div>
          <div class='value'>{rf_label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Compute & show recurrence likelihoods (PERCENTAGES) ---
rec7, rec30, risk7, risk30 = compute_likelihoods(forecast)

cols_bottom = st.columns(3)
with cols_bottom[0]:
    pct_display = f"{rec7:.0f}%" if rec7 is not None and not pd.isna(rec7) else "—"
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>Likelihood (Next 7 Days)</div>
          <div class='value'>{pct_display}</div>
          <div>{likelihood_badge(rec7)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with cols_bottom[1]:
    pct_display = f"{rec30:.0f}%" if rec30 is not None and not pd.isna(rec30) else "—"
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>Likelihood (Next 30 Days)</div>
          <div class='value'>{pct_display}</div>
          <div>{likelihood_badge(rec30)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with cols_bottom[2]:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>Risk Days (7 / 30)</div>
          <div class='value'>{risk7}/{risk30}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# === Charts ===
st.markdown("<br>", unsafe_allow_html=True)
if not env.empty and "TIME" in env.columns and "CHL" in env.columns:
    env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce")
    env = env.dropna(subset=["TIME"])
    st.plotly_chart(
        px.line(
            env, x="TIME", y="CHL",
            title="CHL – 30-Day Environmental History",
            color_discrete_sequence=[PRIMARY_GRAD_2],
            template=PLOTLY_TEMPLATE
        ),
        use_container_width=True
    )
elif not forecast.empty and "date" in forecast.columns and "predicted_chl" in forecast.columns:
    st.plotly_chart(
        px.scatter(
            forecast, x="date", y="predicted_chl",
            title="Predicted CHL (forecast log)",
            color_discrete_sequence=[PRIMARY_GRAD_1],
            template=PLOTLY_TEMPLATE
        ),
        use_container_width=True
    )
else:
    st.info("No environmental or forecast data found for this region yet.")

# === Footer ===
st.markdown(
    f"""
    <hr style='border:0;height:2px;background:linear-gradient(90deg,{PRIMARY_GRAD_1},{PRIMARY_GRAD_2});opacity:.7;'>
    <div style='text-align:center;color:#A9D6E5;font-size:12px;'>
      © {datetime.now().year} MARS • Marine Autonomous Risk System
    </div>
    """,
    unsafe_allow_html=True
)
