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
body {{
  background: radial-gradient(circle at 20% 20%, {PRIMARY_DARK} 0%, #001628 100%);
  color: #FFFFFF;
}}
.block-container {{
  background: transparent !important;
  padding-top: 0rem !important;
}}
.marine-hero {{
  background: linear-gradient(90deg, {PRIMARY_GRAD_1}, {PRIMARY_GRAD_2});
  color: white; border-radius: 20px;
  padding: 24px 28px; box-shadow: 0 10px 25px rgba(0,0,0,.4);
}}
.kpi {{
  background: rgba(255,255,255,.1);
  border-radius: 16px;
  padding: 14px 18px;
  box-shadow: inset 0 0 12px rgba(255,255,255,.05);
}}
.kpi .label {{ color: #A9D6E5; font-size: 13px; letter-spacing: .3px; }}
.kpi .value {{ font-size: 24px; font-weight: 700; color: #FFFFFF; }}
.badge {{
  display:inline-block;padding:5px 10px;border-radius:20px;font-size:12px;font-weight:600;
}}
.badge.low {{ background: rgba(42,157,143,.25);color:{GREEN}; }}
.badge.med {{ background: rgba(255,183,3,.25);color:{AMBER}; }}
.badge.high {{ background: rgba(230,57,70,.25);color:{RED}; }}
.soft-card {{
  background: rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.1);
  border-radius:20px;padding:16px;
  box-shadow:0 6px 16px rgba(0,0,0,.3);
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
    "peiraeus": {"title": "Piraeus (Greece)", "bbox": (37.9, 38.1, 23.5, 23.8), "color": "#FF6B6B"},
    "limassol": {"title": "Limassol (Cyprus)", "bbox": (34.6, 34.8, 33.0, 33.2), "color": "#00B894"},
}

# === Helper functions ===
def _list_files(): return sorted(os.listdir(data_dir))

def latest_env_file(region):
    files = [f for f in _list_files() if re.match(rf"env_history_{region}_.+\\.csv$", f, flags=re.IGNORECASE)]
    return os.path.join(data_dir, sorted(files)[-1]) if files else None

def load_forecast(region):
    for name in [f"forecast_log_{region}.csv", f"forecast_{region}.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
    return pd.DataFrame()

def load_env(region):
    f = latest_env_file(region)
    if not f: return pd.DataFrame()
    df = pd.read_csv(f)
    df.columns = [c.strip().upper() for c in df.columns]
    # auto-fix lowercase or DATE columns
    if "TIME" not in df.columns:
        for alias in ["time","date","datetime","ts"]:
            if alias.upper() in df.columns or alias in df.columns:
                df = df.rename(columns={alias.upper(): "TIME"} if alias.upper() in df.columns else {alias: "TIME"})
                break
    return df

def likelihood_badge(p):
    if p is None: return "<span class='badge'>—</span>"
    if p <= 20: return f\"<span class='badge low'>{p:.0f}% • Low</span>\"
    elif p <= 60: return f\"<span class='badge med'>{p:.0f}% • Moderate</span>\"
    return f\"<span class='badge high'>{p:.0f}% • High</span>\"

# === Map ===
region = st.session_state.get("region","thermaikos")
lat_center = np.mean([REGIONS[r]["bbox"][0]+REGIONS[r]["bbox"][1] for r in REGIONS])/2
lon_center = np.mean([REGIONS[r]["bbox"][2]+REGIONS[r]["bbox"][3] for r in REGIONS])/2

m = folium.Map(location=[lat_center, lon_center], zoom_start=6, tiles="cartodbpositron")
for key,val in REGIONS.items():
    lat_min,lat_max,lon_min,lon_max = val["bbox"]
    folium.Rectangle(bounds=[[lat_min,lon_min],[lat_max,lon_max]],color=val["color"],fill=True,
                     fill_opacity=0.4,popup=val["title"]).add_to(m)

st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#A9D6E5;font-weight:600;'>📍 Click a region below</div>",unsafe_allow_html=True)
click = st_folium(m,height=600,width=1400,key="mars_map")  # Bigger map

if click and click.get("last_clicked"):
    lat,lon = click["last_clicked"]["lat"],click["last_clicked"]["lng"]
    for k,v in REGIONS.items():
        lat_min,lat_max,lon_min,lon_max=v["bbox"]
        if lat_min<=lat<=lat_max and lon_min<=lon<=lon_max:
            region=k
st.session_state.region=region
region_title=REGIONS[region]["title"]

# === Data load ===
forecast, env = load_forecast(region), load_env(region)

# === KPIs ===
st.markdown(f"<h3 style='color:#A9D6E5;margin-top:1em;'>{region_title}</h3>", unsafe_allow_html=True)
cols=st.columns(3)
with cols[0]:
    val=forecast["predicted_chl"].iloc[-1] if not forecast.empty else None
    st.markdown(f\"<div class='kpi'><div class='label'>Predicted CHL</div><div class='value'>{val:.3f if val else '—'} mg/m³</div></div>\",unsafe_allow_html=True)
with cols[1]:
    thresh=forecast["threshold_used"].iloc[-1] if "threshold_used" in forecast else None
    st.markdown(f\"<div class='kpi'><div class='label'>Threshold</div><div class='value'>{thresh:.3f if thresh else '—'}</div></div>\",unsafe_allow_html=True)
with cols[2]:
    risk_flag=forecast["bloom_risk_flag"].iloc[-1] if "bloom_risk_flag" in forecast else None
    st.markdown(f\"<div class='kpi'><div class='label'>Bloom Today</div><div class='value'>{'Yes' if risk_flag==1 else 'No' if risk_flag==0 else '—'}</div></div>\",unsafe_allow_html=True)

# === Charts ===
st.markdown("<br>", unsafe_allow_html=True)
if not env.empty and "TIME" in env.columns and "CHL" in env.columns:
    env["TIME"]=pd.to_datetime(env["TIME"],errors="coerce")
    st.plotly_chart(px.line(env,x="TIME",y="CHL",title="CHL – 30-Day Environmental History",
                            color_discrete_sequence=[PRIMARY_GRAD_2],template=PLOTLY_TEMPLATE),use_container_width=True)
elif not forecast.empty:
    st.plotly_chart(px.scatter(forecast,x="date",y="predicted_chl",title="Predicted CHL (forecast log)",
                               color_discrete_sequence=[PRIMARY_GRAD_1],template=PLOTLY_TEMPLATE),use_container_width=True)
else:
    st.info("No environmental or forecast data found for this region yet.")

# === Footer ===
st.markdown(f\"\"\"<hr style='border:0;height:2px;background:linear-gradient(90deg,{PRIMARY_GRAD_1},{PRIMARY_GRAD_2});opacity:.7;'>
<div style='text-align:center;color:#A9D6E5;font-size:12px;'>© {datetime.now().year} MARS • Marine Autonomous Risk System</div>\"\"\",unsafe_allow_html=True)
