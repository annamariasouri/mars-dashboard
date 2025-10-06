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

# === Interactive Map ===
m = folium.Map(location=[38.0, 24.0], zoom_start=6, tiles="CartoDB positron")

for region, info in REGIONS.items():
    df = load_forecast(region)
    if df is not None:
        latest = df.iloc[-1]
        risk = latest["bloom_risk_flag"]
        color = "red" if risk else "green"
        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=f"{info['name']} — CHL: {latest['predicted_chl']:.2f} mg/m³",
            tooltip=info["name"],
            icon=folium.Icon(color=color)
        ).add_to(m)

st.markdown("### 🗺️ Click markers for bloom info")
st_data = st_folium(m, width=800, height=500)

# === Detail panel for selected region ===
if st_data and st_data.get("last_object_clicked_tooltip"):
    selected = st_data["last_object_clicked_tooltip"]
    region_id = [k for k, v in REGIONS.items() if v["name"] == selected][0]
    df_selected = load_forecast(region_id)

    if df_selected is not None:
        latest = df_selected.iloc[-1]

        st.markdown(f"## 📍 {selected} — Latest Forecast ({latest['date'].date()})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg CHL", f"{latest['predicted_chl']:.2f} mg/m³")
        col2.metric("Bloom Risk", "⚠️ Yes" if latest["bloom_risk_flag"] else "✅ No")
        col3.metric("Threshold", f"{latest['threshold_used']:.2f} mg/m³")

        st.markdown("### 📈 CHL Forecast Trend")
        st.line_chart(df_selected.set_index("date")["predicted_chl"])

        st.markdown("### 📊 Summary")
        col4, col5, col6 = st.columns(3)
        col4.metric("Max CHL", f"{df_selected['predicted_chl'].max():.2f}")
        col5.metric("Min CHL", f"{df_selected['predicted_chl'].min():.2f}")
        col6.metric("Bloom Days", int(df_selected["bloom_risk_flag"].sum()))
