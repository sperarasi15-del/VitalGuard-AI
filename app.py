
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from model import detect_anomalies
from lstm_model import (
    prepare_time_series,
    train_lstm_autoencoder,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VitalGuard AI",
    page_icon="🏥",
    layout="wide",
)

VITAL_SIGNS = [
    "HR",
    "RespRate",
    "SaO2",
    "Temp",
    "SysABP",
    "DiasABP",
    "NISysABP",
    "NIDiasABP",
    "NIMAP",
]

PATIENT_FILES = {
    "P001": "deploy_data/132539.txt",
    "P002": "deploy_data/132540.txt",
    "P003": "deploy_data/132541.txt",
}


# ============================================================
# HEADER
# ============================================================

st.markdown("# 🏥 VitalGuard AI")
st.subheader(
    "Intelligent Early-Warning System for Patient Vital Patterns"
)
st.caption("AI-powered • Data-driven • Explainable")
st.markdown("---")


# ============================================================
# WELCOME
# ============================================================

st.header("👋 Welcome to VitalGuard AI")

st.write(
    """
    VitalGuard AI is a research prototype that analyses ICU
    physiological time-series data and identifies unusual
    patterns using statistical and temporal machine-learning
    techniques.
    """
)

st.info(
    "⚕️ Research and educational prototype only. "
    "It does not provide a medical diagnosis or replace "
    "qualified healthcare professionals."
)


# ============================================================
# REFERENCE VITAL SIGNS
# ============================================================

st.markdown("---")
st.header("❤️ Reference Vital Signs")

st.write(
    "General adult reference values are provided for "
    "educational context. They are not the AI model's "
    "anomaly-detection thresholds."
)

reference_data = pd.DataFrame({
    "Vital Sign": [
        "Heart Rate (HR)",
        "Respiratory Rate (RR)",
        "SpO₂",
        "Temperature",
        "Blood Pressure",
    ],
    "General Adult Reference": [
        "60–100 bpm",
        "12–20 breaths/min",
        "95–100%",
        "Approximately 36.1–37.2 °C",
        "Approximately 90/60–120/80 mmHg",
    ],
})

st.dataframe(
    reference_data,
    hide_index=True,
    width="stretch",
)

st.caption(
    "Reference values are general educational information, "
    "not individual medical advice."
)


# ============================================================
# PATIENT DATA SELECTION
# ============================================================

st.markdown("---")
st.header("📤 Patient Data")

col1, col2 = st.columns(2)

with col1:
    selected_patient = st.selectbox(
        "Select Demonstration Patient",
        list(PATIENT_FILES.keys()),
    )

with col2:
    uploaded_file = st.file_uploader(
        "Or upload patient data (.txt or .csv)",
        type=["txt", "csv"],
    )


# ============================================================
# LOAD DATA
# ============================================================

uploaded_temp_path = None

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".txt",
            delete=False,
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            uploaded_temp_path = temp_file.name

        file_path = uploaded_temp_path
        patient_display_name = "Uploaded Patient"

    except Exception as error:
        st.error(f"Unable to prepare uploaded file: {error}")
        st.stop()

else:
    file_path = PATIENT_FILES[selected_patient]
    patient_display_name = selected_patient

if not os.path.isfile(file_path):
    st.error(
        f"Patient file not found: {file_path}. "
        "Check that the deploy_data folder is present."
    )
    st.stop()

try:
    raw_data = pd.read_csv(file_path)
except Exception as error:
    st.error(f"Unable to read patient data: {error}")
    st.stop()

required_columns = ["Time", "Parameter", "Value"]
missing_columns = [
    column for column in required_columns
    if column not in raw_data.columns
]

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown("---")
st.header("👤 Patient Information")

record_rows = raw_data[
    raw_data["Parameter"].astype(str) == "RecordID"
]

record_id = (
    record_rows["Value"].iloc[0]
    if not record_rows.empty
    else "Unknown"
)

info1, info2, info3 = st.columns(3)

info1.metric("Patient", patient_display_name)
info2.metric("Record ID", str(record_id))
info3.metric("Data Points", len(raw_data))

with st.expander("📋 View Raw Patient Data"):
    st.dataframe(
        raw_data,
        hide_index=True,
        width="stretch",
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("---")

analyze_patient = st.button(
    "🔍 Analyze Patient",
    type="primary",
    width="stretch",
)


# ============================================================
# MAIN ANALYSIS
# All analysis results must remain inside this block.
# ============================================================

if analyze_patient:

    try:
        # ----------------------------------------------------
        # ISOLATION FOREST
        # ----------------------------------------------------

        with st.spinner(
            "Running Isolation Forest anomaly detection..."
        ):
            time_series = detect_anomalies(file_path)

        if time_series.empty:
            st.warning("No usable time-series observations found.")
            st.stop()

        available_vitals = [
            vital for vital in VITAL_SIGNS
            if vital in time_series.columns
        ]

        if not available_vitals:
            st.warning("No supported vital signs were found.")
            st.stop()

        anomaly_data = time_series[
            time_series["Anomaly"] == -1
        ]

        normal_data = time_series[
            time_series["Anomaly"] == 1
        ]

        total_points = len(time_series)
        anomaly_count = len(anomaly_data)
        normal_count = len(normal_data)

        anomaly_rate = (
            anomaly_count / total_points * 100
            if total_points else 0
        )

        # ----------------------------------------------------
        # ISOLATION FOREST RESULTS
        # ----------------------------------------------------

        st.markdown("---")
        st.header("🌲 Isolation Forest Anomaly Detection")

        result1, result2, result3 = st.columns(3)

        result1.metric("Total Observations", total_points)
        result2.metric("Detected Anomalies", anomaly_count)
        result3.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

        if anomaly_count == 0:
            st.success("No unusual patterns were detected.")
        else:
            st.warning(
                f"The model flagged {anomaly_count} unusual "
                "observation(s). These require contextual review."
            )

            st.subheader("⏱️ Detected Anomaly Periods")
            st.dataframe(
                anomaly_data,
                width="stretch",
            )

        # ----------------------------------------------------
        # PERSONALIZED PATIENT BASELINE
        # ----------------------------------------------------

        st.markdown("---")
        st.header("📊 Personalized Patient Baseline")

        baseline_data = time_series[available_vitals]

        baseline = baseline_data.median()
        spread = baseline_data.std().replace(0, 1)

        baseline_table = pd.DataFrame({
            "Vital Sign": available_vitals,
            "Patient Baseline": [
                baseline[vital] for vital in available_vitals
            ],
            "Standard Deviation": [
                spread[vital] for vital in available_vitals
            ],
        })

        st.dataframe(
            baseline_table.round(2),
            hide_index=True,
            width="stretch",
        )

        st.info(
            "This baseline is calculated from the selected "
            "patient's observed data. It is not a clinical "
            "reference range."
        )

        # ----------------------------------------------------
        # EXPLAINABLE ANOMALY RESULTS
        # ----------------------------------------------------

        st.markdown("---")
        st.header("🔎 Explainable Anomaly Results")

        if anomaly_count > 0:
            explanations = []

            for timestamp in anomaly_data.index:
                if timestamp not in baseline_data.index:
                    continue

                observed = baseline_data.loc[timestamp]

                deviations = (
                    (observed - baseline) / spread
                ).abs().sort_values(ascending=False)

                top_vitals = deviations.head(3)

                explanation = ", ".join(
                    f"{vital} ({value:.2f}σ)"
                    for vital, value in top_vitals.items()
                )

                explanations.append({
                    "Time": str(timestamp),
                    "Largest Observed Deviations": explanation,
                })

            if explanations:
                st.dataframe(
                    pd.DataFrame(explanations),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("No explanation details are available.")

            st.caption(
                "These are statistical deviations from the "
                "patient's observed baseline, not a diagnosis."
            )
        else:
            st.success(
                "There are no Isolation Forest anomaly periods "
                "to explain for this analysis."
            )

        # ----------------------------------------------------
        # LSTM AUTOENCODER
        # ----------------------------------------------------

        st.markdown("---")
        st.header("🧠 LSTM Autoencoder Analysis")

        lstm_result = None

        try:
            lstm_data = prepare_time_series(file_path)

            if len(lstm_data) < 3:
                st.warning(
                    "Not enough observations for LSTM analysis."
                )
            else:
                with st.spinner(
                    "Training LSTM Autoencoder..."
                ):
                    (
                        lstm_result,
                        trained_lstm,
                        lstm_scaler,
                        lstm_threshold,
                    ) = train_lstm_autoencoder(
                        lstm_data,
                        epochs=30,
                    )

                lstm_anomalies = lstm_result[
                    lstm_result["LSTM_Anomaly"]
                ]

                lstm_col1, lstm_col2 = st.columns(2)

                lstm_col1.metric(
                    "LSTM Anomalies",
                    len(lstm_anomalies),
                )
                lstm_col2.metric(
                    "Reconstruction Threshold",
                    f"{lstm_threshold:.4f}",
                )

                if not lstm_anomalies.empty:
                    st.subheader("⏱️ LSTM Detected Periods")
                    st.dataframe(
                        lstm_anomalies,
                        width="stretch",
                    )
                else:
                    st.success(
                        "No LSTM anomalies were detected."
                    )

        except Exception as error:
            st.error(f"LSTM analysis failed: {error}")

        # ----------------------------------------------------
        # HYBRID AI
        # ----------------------------------------------------

        st.markdown("---")
        st.header("🤖 Hybrid AI Analysis")

        hybrid = time_series.copy()
        hybrid["IsolationForest"] = (
            hybrid["Anomaly"] == -1
        )
        hybrid["LSTM"] = False

        if lstm_result is not None:
            common_times = hybrid.index.intersection(
                lstm_result.index
            )

            hybrid.loc[common_times, "LSTM"] = (
                lstm_result.loc[
                    common_times, "LSTM_Anomaly"
                ].astype(bool)
            )

            hybrid["Hybrid_Anomaly"] = (
                hybrid["IsolationForest"] | hybrid["LSTM"]
            )

            hybrid_col1, hybrid_col2, hybrid_col3 = (
                st.columns(3)
            )

            hybrid_col1.metric(
                "Isolation Forest",
                int(hybrid["IsolationForest"].sum()),
            )
            hybrid_col2.metric(
                "LSTM",
                int(hybrid["LSTM"].sum()),
            )
            hybrid_col3.metric(
                "Hybrid AI",
                int(hybrid["Hybrid_Anomaly"].sum()),
            )

            st.info(
                "Hybrid AI flags observations identified by "
                "either Isolation Forest or the LSTM Autoencoder. "
                "The combined count may include overlapping flags."
            )

        else:
            st.info(
                "Hybrid results require a successful LSTM analysis."
            )

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        st.markdown("---")
        st.header("📊 Model Comparison")

        st.write(
            """
            The following comparison summarizes the detected
            anomaly counts for this patient. Counts alone do not
            establish which model is more accurate.
            """
        )

        comparison_rows = [
            {
                "Model": "Isolation Forest",
                "Detected Anomalies": anomaly_count,
            }
        ]

        if lstm_result is not None:
            comparison_rows.append({
                "Model": "LSTM Autoencoder",
                "Detected Anomalies": int(
                    lstm_result["LSTM_Anomaly"].sum()
                ),
            })

            comparison_rows.append({
                "Model": "Hybrid AI",
                "Detected Anomalies": int(
                    hybrid["Hybrid_Anomaly"].sum()
                ),
            })

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df,
            hide_index=True,
            width="stretch",
        )

        comparison_fig = go.Figure()

        comparison_fig.add_trace(
            go.Bar(
                x=comparison_df["Model"],
                y=comparison_df["Detected Anomalies"],
                name="Detected Anomalies",
            )
        )

        comparison_fig.update_layout(
            title="Detected Anomaly Count by Model",
            xaxis_title="Model",
            yaxis_title="Number of flagged observations",
            height=400,
        )

        st.plotly_chart(
            comparison_fig,
            width="stretch",
        )

        st.warning(
            "This count comparison is descriptive, not an accuracy "
            "benchmark. Precision, recall, F1-score and ROC-AUC "
            "require an appropriate labelled evaluation dataset."
        )

        # ----------------------------------------------------
        # INTERACTIVE VITAL DASHBOARD
        # ----------------------------------------------------

        st.markdown("---")
        st.header("📈 Interactive Patient Vital Dashboard")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            selected_vital = st.selectbox(
                "Select Vital Sign",
                available_vitals,
            )

        with chart_col2:
            show_anomalies = st.checkbox(
                "Show Isolation Forest anomalies",
                value=True,
            )

        chart_data = time_series[[selected_vital]].dropna()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[selected_vital],
                mode="lines+markers",
                name=selected_vital,
                hovertemplate=(
                    "<b>Time:</b> %{x}<br>"
                    f"<b>{selected_vital}:</b> "
                    "%{y:.2f}<extra></extra>"
                ),
            )
        )

        if show_anomalies and anomaly_count > 0:
            anomaly_points = chart_data.loc[
                chart_data.index.isin(anomaly_data.index)
            ]

            if not anomaly_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points.index,
                        y=anomaly_points[selected_vital],
                        mode="markers",
                        name="Anomaly",
                        marker={"size": 12, "symbol": "x"},
                    )
                )

        fig.update_layout(
            title=(
                f"{selected_vital} Timeline — "
                f"{patient_display_name}"
            ),
            xaxis_title="Time",
            yaxis_title=selected_vital,
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.subheader("📊 Vital Sign Summary")

        summary1, summary2, summary3 = st.columns(3)

        summary1.metric(
            "Average",
            f"{chart_data[selected_vital].mean():.2f}",
        )
        summary2.metric(
            "Minimum",
            f"{chart_data[selected_vital].min():.2f}",
        )
        summary3.metric(
            "Maximum",
            f"{chart_data[selected_vital].max():.2f}",
        )

        # ----------------------------------------------------
        # AI INTERPRETATION
        # ----------------------------------------------------

        st.markdown("---")
        st.header("🧠 AI Interpretation")

        if anomaly_count == 0:
            st.success(
                "The Isolation Forest model did not flag unusual "
                "observations in this analysis."
            )
        else:
            st.warning(
                f"The model flagged {anomaly_count} unusual "
                f"observation(s) out of {total_points}."
            )

        st.write(
            "The results identify statistical patterns in the "
            "recorded data. They do not determine the cause of "
            "a change or provide a medical diagnosis."
        )

    except Exception as error:
        st.error(f"Patient analysis failed: {error}")

    finally:
        # Delete temporary uploaded data after analysis.
        if uploaded_temp_path and os.path.exists(uploaded_temp_path):
            try:
                os.unlink(uploaded_temp_path)
            except OSError:
                pass



# --------------------------------------------
# SDG 3 — GOOD HEALTH AND WELL-BEING
# --------------------------------------------

st.markdown("---")
st.header("🌍 SDG 3 — Good Health and Well-being")

st.write(
    "VitalGuard AI explores how machine learning can identify "
    "unusual patterns in ICU vital-sign data for further review."
)

st.subheader("📊 Measurable Project Outcomes")

tab1, tab2 = st.tabs(["This app: synthetic-anomaly benchmark (PhysioNet 2012)",
                      "Round 3 case study (MIMIC-III, simulated events)"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("LSTM recall (3 patients)", "1.00")
    c2.metric("LSTM best F1", "0.46")
    c3.metric("Isolation Forest recall", "0.00 – 0.33")
    st.caption("Controlled synthetic anomalies, not clinical events.")

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("False-alarm rate", "2.51%", "-33% vs Z-score")
    c2.metric("Precision", "77.8%")
    c3.metric("ROC-AUC", "0.979")
    st.caption("One MIMIC-III record with two simulated events; not clinical validation.")
    
st.subheader("Potential Societal Benefits")

st.markdown(
    """
    - **Earlier review:** highlights unusual observations for further
      investigation by healthcare professionals.
    - **Explainability:** identifies vital signs that deviate from
      the patient's observed baseline.
    - **Monitoring research:** helps evaluate machine-learning
      approaches to physiological time-series.
    - **Responsible AI:** communicates model limitations and avoids
      treating anomaly flags as diagnoses.
    """
)

st.warning(
    "Clinical validation has not been established. These results "
    "do not prove reduced false alarms, improved patient outcomes, "
    "or suitability for clinical use."
)

st.caption(
    "Dataset: PhysioNet Challenge 2012 sample records | "
    "Models: Isolation Forest and LSTM Autoencoder | SDG 3"
)
