import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------
# ✅ Simulate sensor data
# -------------------------------------------------------
def simulate_sensor_data(n_samples=200):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq="min")

    sensor_1 = np.random.normal(50, 5, n_samples)
    sensor_2 = np.random.normal(70, 7, n_samples)

    # Inject visible anomalies
    anomalies = np.random.choice(n_samples, size=6, replace=False)
    sensor_1[anomalies] += np.random.normal(40, 10, 6)
    sensor_2[anomalies] += np.random.normal(55, 12, 6)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_1": sensor_1,
        "sensor_2": sensor_2
    })

    return data


# -------------------------------------------------------
# ✅ Streamlit UI Setup
# -------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🛠️ AI-Powered Predictive Maintenance Dashboard")
st.markdown("Simulated sensor data with anomaly detection using Isolation Forest.")


# Load simulated data
data = simulate_sensor_data()

# Sidebar settings
st.sidebar.header("⚙️ Configuration")
contamination = st.sidebar.slider("Anomaly contamination level", 0.01, 0.15, 0.05)


# -------------------------------------------------------
# ✅ Train Isolation Forest
# -------------------------------------------------------
features = ["sensor_1", "sensor_2"]
X = data[features]

model = IsolationForest(contamination=contamination, random_state=42)
data["anomaly_label"] = model.fit_predict(X)
data["anomaly_label"] = data["anomaly_label"].map({1: "Normal", -1: "Anomaly"})


# Ensure numeric dtypes (critical fix)
data["sensor_1"] = pd.to_numeric(data["sensor_1"], errors="coerce")
data["sensor_2"] = pd.to_numeric(data["sensor_2"], errors="coerce")

# Split data for plotting
norm = data[data["anomaly_label"] == "Normal"]
anom = data[data["anomaly_label"] == "Anomaly"]


# -------------------------------------------------------
# ✅ Sensor Trends Over Time
# -------------------------------------------------------
st.subheader("📉 Real-Time Sensor Readings")

fig1 = px.line(data, x="timestamp", y=["sensor_1", "sensor_2"], title="Sensor Trends Over Time")
st.plotly_chart(fig1, use_container_width=True)


# -------------------------------------------------------
# ✅ Scatter Plot with FIXED AXES + VISIBLE POINTS
# -------------------------------------------------------
st.subheader("🚨 Anomaly Detection")

# Auto-calc ranges so Plotly NEVER zooms incorrectly
pad_x = (data["sensor_1"].max() - data["sensor_1"].min()) * 0.2 or 10
pad_y = (data["sensor_2"].max() - data["sensor_2"].min()) * 0.2 or 10

x_range = [data["sensor_1"].min() - pad_x, data["sensor_1"].max() + pad_x]
y_range = [data["sensor_2"].min() - pad_y, data["sensor_2"].max() + pad_y]

fig2 = go.Figure()

# ✅ Normal points
fig2.add_trace(go.Scattergl(
    x=norm["sensor_1"], y=norm["sensor_2"],
    mode="markers",
    name="Normal",
    marker=dict(size=8, symbol="circle", color="#0094b8", opacity=0.85)
))

# ✅ Anomaly points (larger + diamond)
fig2.add_trace(go.Scattergl(
    x=anom["sensor_1"], y=anom["sensor_2"],
    mode="markers",
    name="Anomaly",
    marker=dict(size=16, symbol="diamond", color="red", line=dict(width=1, color="black"))
))

fig2.update_layout(
    title="Sensor Anomalies",
    xaxis_title="sensor_1",
    yaxis_title="sensor_2",
    legend_title_text="Label",
)

# ✅ Force Plotly to show real numeric ranges
fig2.update_xaxes(range=x_range)
fig2.update_yaxes(range=y_range)

st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------------
# ✅ Insights Panel
# -------------------------------------------------------
num_anomalies = len(anom)
st.metric("⚠️ Detected Anomalies", value=num_anomalies)

if num_anomalies > 0:
    st.warning("⚠️ Maintenance required. Anomalous behavior detected.")
else:
    st.success("✅ All systems operating normally.")


# -------------------------------------------------------
# ✅ Raw Data Viewer
# -------------------------------------------------------
with st.expander("🗃 View Raw Sensor Data"):
    st.dataframe(data)


# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Fixed & Optimized Version ✅")
