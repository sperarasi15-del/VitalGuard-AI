import streamlit as st
import matplotlib.pyplot as plt
from model import detect_anomalies

st.set_page_config(
    page_title="VitalGuard AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 VitalGuard AI")
st.subheader("AI-Powered Patient Vital Anomaly Detection")

st.caption(
    "Machine-learning prototype for detecting unusual patterns "
    "in physiological time-series data"
)

st.write(
    "An intelligent system for detecting unusual patterns "
    "in physiological time-series data."
)

# Patient selection
st.sidebar.header("Patient Information")

patient_files = {
    "P001": "deploy_data/132539.txt",
    "P002": "deploy_data/132540.txt",
    "P003": "deploy_data/132541.txt"
}

patient_id = st.sidebar.selectbox(
    "Select Patient",
    list(patient_files.keys())
)

st.sidebar.write("Selected Patient:", patient_id)
st.sidebar.success(
    f"Patient {patient_id} is being monitored by the AI system."
)

# Patient data file
file_path = patient_files[patient_id]

# Run AI anomaly detection
data = detect_anomalies(file_path)

# Get latest measurements
latest = data.iloc[-1]
st.markdown("### 👤 Patient Monitoring")

st.write(
    f"Currently monitoring **{patient_id}** using physiological "
    "time-series data from the PhysioNet dataset."
)

# Display current vital signs
st.markdown("### 📊 Current Vital Signs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "❤️ Heart Rate",
        f"{latest['HR']:.0f} bpm"
    )

with col2:
    if "RespRate" in data.columns:
        st.metric(
            "🫁 Respiratory Rate",
            f"{latest['RespRate']:.0f} /min"
        )
    else:
        st.metric(
            "🫁 Respiratory Rate",
            "Not available"
        )

with col3:
    if "Temp" in data.columns:
        st.metric(
            "🌡️ Temperature",
            f"{latest['Temp']:.1f} °C"
        )
    else:
        st.metric(
            "🌡️ Temperature",
            "Not available"
        )

with col4:
    if "NIMAP" in data.columns:
        st.metric(
            "🩸 NIBP",
            f"{latest['NIMAP']:.0f}"
        )
    else:
        st.metric(
            "🩸 NIBP",
            "Not available"
        )

# Count anomalies
anomaly_count = (data["Anomaly"] == -1).sum()

normal_count = (data["Anomaly"] == 1).sum()

total_count = len(data)

anomaly_percentage = (anomaly_count / total_count) * 100

st.info(
    "🤖 AI Model: Isolation Forest\n\n"
    "The model analyzes physiological measurements and "
    "identifies observations that differ from normal patterns."
)
st.markdown("### 🚨 AI Anomaly Detection")

st.info(
    "🧠 The Isolation Forest model analyzes the patient's "
    "physiological measurements and identifies patterns that "
    "differ significantly from the patient's observed data."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🚨 Anomalies",
        anomaly_count
    )

with col2:
    st.metric(
        "🟢 Normal",
        normal_count
    )

with col3:
    st.metric(
        "📊 Anomaly Rate",
        f"{anomaly_percentage:.1f}%"
    )

if anomaly_count > 0:
    st.warning(
        f"⚠️ AI detected {anomaly_count} unusual time points "
        "in this patient's vital-sign data."
    )
else:
    st.success("🟢 No unusual patterns detected.")

# Show anomaly table
st.markdown("### 🔎 Detected Anomalies")

anomalies = data[data["Anomaly"] == -1]

if len(anomalies) > 0:

    st.write("The AI detected unusual patterns at these time points:")

    for time in anomalies.index:
        st.write(f"🔴 {time}")

    st.dataframe(anomalies)

else:
    st.success("🟢 No anomalies detected.")

st.info(
    "⚠️ This application is a research prototype and is not "
    "intended to provide medical diagnosis."
)

# Heart Rate Anomaly Graph

st.markdown("### 📈 Heart Rate Monitoring")

fig, ax = plt.subplots(figsize=(10, 5))

# Plot all heart-rate measurements
ax.plot(
    data.index,
    data["HR"],
    marker="o",
    label="Heart Rate"
)

# Get anomaly points
anomalies = data[data["Anomaly"] == -1]

# Highlight anomaly points
ax.scatter(
    anomalies.index,
    anomalies["HR"],
    marker="x",
    s=120,
    label="AI Detected Anomaly"
)

ax.set_xlabel("Time")
ax.set_ylabel("Heart Rate (bpm)")
ax.set_title(
    f"Heart Rate Anomaly Detection - {patient_id}"
)

ax.tick_params(axis="x", rotation=45)

ax.legend()
ax.grid(True)

st.pyplot(fig)
st.markdown("---")

st.markdown("### 🌍 SDG 3 — Good Health & Well-being")

st.write(
    "VitalGuard AI supports SDG 3 by using machine learning "
    "to identify unusual patterns in physiological time-series "
    "data. The system is designed as a research prototype to "
    "support data-driven health monitoring."
)

st.caption(
    "Research prototype — anomaly detection only. "
    "Not intended for medical diagnosis."
)

# Normal measurements
normal = data[data["Anomaly"] == 1]

ax.plot(
    normal.index,
    normal["HR"],
    marker="o",
    label="Normal"
)

# Anomaly measurements
anomalies = data[data["Anomaly"] == -1]

ax.scatter(
    anomalies.index,
    anomalies["HR"],
    marker="x",
    s=100,
    label="Anomaly"
)

ax.set_xlabel("Time")
ax.set_ylabel("Heart Rate (bpm)")
ax.set_title("AI-Based Heart Rate Anomaly Detection")

ax.tick_params(axis="x", rotation=45)

ax.legend()
ax.grid(True)

st.pyplot(fig)

