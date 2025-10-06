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
path = f"forecast_log_{thermaikos}.csv"
if os.path.exists(path):
df = pd.read_csv(path, parse_dates=["date"])
df = df.sort_values("date")
return df
else:
return None

# === Page layout ===

st.set_page_config(page_title="MARS Dashboard", layout="wide")
st.markdown(""" <style>
.main {
background-color: #f2f9ff;
} </style>
""", unsafe_allow_html=True)

st.title("🌊 MARS — Marine Autonomous Risk System")
st.caption("Part of ongoing PhD research on real-time marine bloom forecasting using Copernicus Marine data.")

# === CHL / Red Tide Explanation ===

with st.expander("ℹ️ What is CHL and Why It Matters?", expanded=True):
st.markdown("""
**Chlorophyll-a (CHL)** is a proxy for phytoplankton levels in the ocean.
When CHL is abnormally high, it can signal **algal blooms**, including harmful red tides.

```
Red tide events can:
- Release toxins harmful to marine life and humans
- Deplete oxygen and lead to fish kills
- Disrupt fisheries, aquaculture, and coastal tourism

The MARS system monitors daily CHL and environmental indicators to forecast bloom risk in coastal areas.
""")
```

# === Interactive Map ===

m = folium.Map(location=[38.0, 24.0], zoom_start=6, tiles="CartoDB positron")

for region, info in REGIONS.items():
df = load_forecast(region)
if df is not None and not df.empty:
latest = df.iloc[-1]
risk = latest["bloom_risk_flag"]
color = "red" if risk else "green"
folium.Marker(
location=[info["lat"], info["lon"]],
popup=f"{info['name']} — CHL: {latest['predicted_chl']:.2f} mg/m³",
tooltip=info["name"],
icon=folium.Icon(color=color)
).add_to(m)

st.markdown("### 📍 Select a location to view detailed risk information")
st_data = st_folium(m, width=1200, height=700)

# === Detail panel for selected region ===

if st_data and st_data.get("last_object_clicked_tooltip"):
selected = st_data["last_object_clicked_tooltip"]
region_id = [k for k, v in REGIONS.items() if v["name"] == selected][0]
df_selected = load_forecast(region_id)

```
if df_selected is not None and not df_selected.empty:
    latest = df_selected.iloc[-1]

    st.markdown(f"## 📍 {selected} — Forecast Summary ({latest['date'].date()})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg CHL", f"{latest['predicted_chl']:.2f} mg/m³")
    col2.metric("Bloom Risk", "⚠️ Yes" if latest["bloom_risk_flag"] else "✅ No")
    col3.metric("Threshold", f"{latest['threshold_used']:.2f} mg/m³")

    st.markdown("### 📈 CHL Forecast Trend")
    st.line_chart(df_selected.set_index("date")["predicted_chl"])

    if "nh4" in df_selected.columns:
        st.markdown("### 🌱 Nutrients (NH₄, NO₃, PO₄)")
        st.line_chart(df_selected.set_index("date")[["nh4", "no3", "po4"]])

    if "thetao" in df_selected.columns:
        st.markdown("### 🌡️ Temperature & Salinity")
        st.line_chart(df_selected.set_index("date")[["thetao", "so"]])

    st.markdown("### 📊 Summary Statistics")
    col4, col5, col6 = st.columns(3)
    col4.metric("Max CHL", f"{df_selected['predicted_chl'].max():.2f}")
    col5.metric("Min CHL", f"{df_selected['predicted_chl'].min():.2f}")
    col6.metric("Bloom Days", int(df_selected["bloom_risk_flag"].sum()))
```


