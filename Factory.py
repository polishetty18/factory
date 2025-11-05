import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

# -------------------------------
# Function to simulate sensor data
# -------------------------------
def simulate_sensor_data(n_samples=200):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq="min")
    # Simulate normal data with small random variations
    sensor_1 = np.random.normal(loc=50, scale=5, size=n_samples)
    sensor_2 = np.random.normal(loc=70, scale=7, size=n_samples)

    # Inject anomalies with a larger deviation to make them more obvious
    anomalies = np.random.choice(n_samples, size=5, replace=False)
    sensor_1[anomalies] += np.random.normal(30, 5, size=5)
    sensor_2[anomalies] += np.random.normal(40, 5, size=5)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_1": sensor_1,
        "sensor_2": sensor_2
    })
    return data

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🛠️ AI-Powered Predictive Maintenance Dashboard")
st.markdown("Simulated sensor data with anomaly detection for early failure prediction.")

# Load simulated data
data = simulate_sensor_data()

# Sidebar - controls
st.sidebar.header("⚙️ Configuration")
contamination = st.sidebar.slider("Anomaly contamination level", 0.01, 0.15, 0.05)
model_type = st.sidebar.selectbox("Anomaly Detection Model", ["Isolation Forest"])

# Preprocess
features = ["sensor_1", "sensor_2"]
X = data[features]

# Model
model = IsolationForest(contamination=contamination, random_state=42)
data["anomaly_label"] = model.fit_predict(X)
data["anomaly_label"] = data["anomaly_label"].map({1: "Normal", -1: "Anomaly"})

# Main display
st.subheader("📉 Real-Time Sensor Readings")
fig1 = px.line(data, x="timestamp", y=["sensor_1", "sensor_2"], title="Sensor Trends Over Time")
st.plotly_chart(fig1, use_container_width=True)

# Anomaly visualization
st.subheader("🚨 Anomaly Detection")
fig2 = px.scatter(
    data,
    x="sensor_1",
    y="sensor_2",
    color="anomaly_label",
    title="Sensor Anomalies",
    color_discrete_map={'Normal': '#0094b8', 'Anomaly': 'red'},
    # Use a different symbol for anomalies to make them more distinct
    symbol="anomaly_label",
    symbol_map={'Normal': 'circle', 'Anomaly': 'diamond'},
    # Set different sizes for normal vs. anomaly points
    size=data["anomaly_label"].apply(lambda x: 12 if x == "Anomaly" else 5)
)
st.plotly_chart(fig2, use_container_width=True)

# Insights
num_anomalies = (data["anomaly_label"] == "Anomaly").sum()
st.metric("⚠️ Detected Anomalies", value=num_anomalies)
if num_anomalies > 0:
    st.warning("⚠️ Maintenance required. Anomalous behavior detected.")
else:
    st.success("✅ All systems operating normally.")

# Show raw data (optional)
with st.expander("🗃 View Raw Sensor Data"):
    st.dataframe(data)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Hackathon Demo")
