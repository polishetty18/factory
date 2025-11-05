import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Simulate sensor data
# -------------------------------
def simulate_sensor_data(n_samples=200):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq="min")
    sensor_1 = np.random.normal(50, 5, n_samples)
    sensor_2 = np.random.normal(70, 7, n_samples)

    # Inject visible anomalies
    anomalies = np.random.choice(n_samples, size=6, replace=False)
    sensor_1[anomalies] += np.random.normal(40, 10, 6)
    sensor_2[anomalies] += np.random.normal(55, 12, 6)

    return pd.DataFrame({"timestamp": timestamps, "sensor_1": sensor_1, "sensor_2": sensor_2})

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🛠️ AI-Powered Predictive Maintenance Dashboard")
st.markdown("Simulated sensor data with anomaly detection using Isolation Forest.")

# Data
data = simulate_sensor_data()

# Sidebar
st.sidebar.header("⚙️ Configuration")
contamination = st.sidebar.slider("Anomaly contamination level", 0.01, 0.15, 0.05)

# -------------------------------
# Model
# -------------------------------
features = ["sensor_1", "sensor_2"]
X = data[features]

model = IsolationForest(contamination=contamination, random_state=42)
labels = model.fit_predict(X)
data["anomaly_label"] = pd.Series(labels).map({1: "Normal", -1: "Anomaly"})

# -------------------------------
# Hardening: force numeric + drop bad rows
# -------------------------------
for col in ["sensor_1", "sensor_2"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["sensor_1", "sensor_2"], inplace=True)

# Split views (still safe if one is empty)
norm = data[data["anomaly_label"] == "Normal"]
anom = data[data["anomaly_label"] == "Anomaly"]

# -------------------------------
# Time series
# -------------------------------
st.subheader("📉 Real-Time Sensor Readings")
fig_ts = px.line(data, x="timestamp", y=["sensor_1", "sensor_2"], title="Sensor Trends Over Time")
st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------------
# Scatter with guaranteed visibility
# -------------------------------
st.subheader("🚨 Anomaly Detection")

# Compute safe axis ranges
def padded_range(series, pct=0.2, minimum_pad=10):
    if series.empty:
        return [0, 1]
    span = series.max() - series.min()
    pad = max(span * pct, minimum_pad)
    return [series.min() - pad, series.max() + pad]

x_range = padded_range(data["sensor_1"])
y_range = padded_range(data["sensor_2"])

fig_sc = go.Figure()

# Normal points
if not norm.empty:
    fig_sc.add_trace(go.Scattergl(
        x=norm["sensor_1"], y=norm["sensor_2"],
        mode="markers",
        name="Normal",
        marker=dict(size=8, symbol="circle", color="#0094b8", opacity=0.85)
    ))

# Anomaly points
if not anom.empty:
    fig_sc.add_trace(go.Scattergl(
        x=anom["sensor_1"], y=anom["sensor_2"],
        mode="markers",
        name="Anomaly",
        marker=dict(size=16, symbol="diamond", color="red", line=dict(width=1, color="black"))
    ))

fig_sc.update_layout(
    title="Sensor Anomalies",
    xaxis_title="sensor_1",
    yaxis_title="sensor_2",
    legend_title_text="Label"
)

# Force linear numeric axes and ranges
fig_sc.update_xaxes(type="linear", range=x_range)
fig_sc.update_yaxes(type="linear", range=y_range)

st.plotly_chart(fig_sc, use_container_width=True)

# -------------------------------
# Insights
# -------------------------------
st.metric("⚠️ Detected Anomalies", value=len(anom))
if len(anom) > 0:
    st.warning("⚠️ Maintenance required. Anomalous behavior detected.")
else:
    st.success("✅ All systems operating normally.")

# -------------------------------
# Debug panel (helpful if chart is empty)
# -------------------------------
with st.expander("🧪 Debug panel"):
    st.write("Dtypes:", data.dtypes)
    st.write("Head:", data.head())
    st.write("Describe:", data[["sensor_1", "sensor_2"]].describe())
    st.write("Counts:", {"norm": len(norm), "anom": len(anom)})

# -------------------------------
# Raw data
# -------------------------------
with st.expander("🗃 View Raw Sensor Data"):
    st.dataframe(data)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Hardened Plotly rendering")
