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

# Main display - keep unchanged
st.subheader("📉 Real-Time Sensor Readings")
fig1 = px.line(data, x="timestamp", y=["sensor_1", "sensor_2"], title="Sensor Trends Over Time")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Minimal, focused fix for Anomaly visualization
# -------------------------------
st.subheader("🚨 Anomaly Detection")

# Force numeric types (minimal hardening, doesn't change other app logic)
data["sensor_1"] = pd.to_numeric(data["sensor_1"], errors="coerce")
data["sensor_2"] = pd.to_numeric(data["sensor_2"], errors="coerce")

# Filter anomalies only (we will plot *only* anomaly points as requested)
anom = data[data["anomaly_label"] == "Anomaly"]

# Compute padding and axis ranges from full data so anomalies are in view
span_x = data["sensor_1"].max() - data["sensor_1"].min()
span_y = data["sensor_2"].max() - data["sensor_2"].min()
pad_x = span_x * 0.1 if span_x > 0 else 5
pad_y = span_y * 0.1 if span_y > 0 else 5
x_range = [data["sensor_1"].min() - pad_x, data["sensor_1"].max() + pad_x]
y_range = [data["sensor_2"].min() - pad_y, data["sensor_2"].max() + pad_y]

# If there are anomalies -> plot them (red diamonds, larger)
if not anom.empty:
    fig2 = px.scatter(
        anom,
        x="sensor_1",
        y="sensor_2",
        title="Sensor Anomalies (only anomaly points shown)"
    )
    # force marker style for anomalies
    fig2.update_traces(marker=dict(size=14, symbol="diamond", color="red",
                                   line=dict(width=1, color="black")))
    # force axis ranges to include anomalies
    fig2.update_xaxes(range=x_range, title_text="sensor_1", type="linear")
    fig2.update_yaxes(range=y_range, title_text="sensor_2", type="linear")
    st.plotly_chart(fig2, use_container_width=True)
else:
    # No anomalies detected: show an empty plot with proper numeric axes and a message
    empty_df = pd.DataFrame({"sensor_1": [data["sensor_1"].min(), data["sensor_1"].max()],
                             "sensor_2": [data["sensor_2"].min(), data["sensor_2"].max()]})
    fig2 = px.scatter(
        empty_df,
        x="sensor_1",
        y="sensor_2",
        title="Sensor Anomalies (no anomalies detected)"
    )
    fig2.update_traces(marker=dict(size=0, opacity=0))  # hide the helper points
    fig2.update_xaxes(range=x_range, title_text="sensor_1", type="linear")
    fig2.update_yaxes(range=y_range, title_text="sensor_2", type="linear")
    # add a center annotation so user sees nothing is present
    fig2.add_annotation(text="No anomalies detected", xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
    st.plotly_chart(fig2, use_container_width=True)

# Insights (unchanged)
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
