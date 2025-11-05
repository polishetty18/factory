import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

# -------------------------------------------------------
# ✅ Simulate sensor data (with stronger anomaly injection)
# -------------------------------------------------------
def simulate_sensor_data(n_samples=200):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq="min")

    sensor_1 = np.random.normal(50, 5, n_samples)
    sensor_2 = np.random.normal(70, 7, n_samples)

    # Inject clear anomalies
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

# Sidebar Config
st.sidebar.header("⚙️ Configuration")
contamination = st.sidebar.slider("Anomaly contamination level", 0.01, 0.15, 0.05)
model_choice = st.sidebar.selectbox("Anomaly Detection Model", ["Isolation Forest"])


# -------------------------------------------------------
# ✅ Train Model
# -------------------------------------------------------
features = ["sensor_1", "sensor_2"]
X = data[features]

model = IsolationForest(contamination=contamination, random_state=42)
data["anomaly_label"] = model.fit_predict(X)
data["anomaly_label"] = data["anomaly_label"].map({1: "Normal", -1: "Anomaly"})

# Create a size column for Plotly (important!)
data["point_size"] = data["anomaly_label"].apply(lambda x: 14 if x == "Anomaly" else 6)


# -------------------------------------------------------
# ✅ Sensor Time-Series Plot
# -------------------------------------------------------
st.subheader("📉 Real-Time Sensor Readings")

fig1 = px.line(
    data,
    x="timestamp",
    y=["sensor_1", "sensor_2"],
    title="Sensor Trends Over Time"
)
st.plotly_chart(fig1, use_container_width=True)


# -------------------------------------------------------
# ✅ Scatterplot with Correct Anomaly Visualization
# -------------------------------------------------------
st.subheader("🚨 Anomaly Detection")

fig2 = px.scatter(
    data,
    x="sensor_1",
    y="sensor_2",
    color="anomaly_label",
    symbol="anomaly_label",
    size="point_size",
    color_discrete_map={"Normal": "#0094b8", "Anomaly": "red"},
    symbol_map={"Normal": "circle", "Anomaly": "diamond"},
    title="Sensor Anomalies"
)

st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------------
# ✅ Insights
# -------------------------------------------------------
num_anomalies = (data["anomaly_label"] == "Anomaly").sum()
st.metric("⚠️ Detected Anomalies", value=num_anomalies)

if num_anomalies > 0:
    st.warning("⚠️ Maintenance required. Anomalous behavior detected.")
else:
    st.success("✅ All systems operating normally.")


# -------------------------------------------------------
# ✅ Raw Data
# -------------------------------------------------------
with st.expander("🗃 View Raw Sensor Data"):
    st.dataframe(data)


# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Enhanced Version ✅")
