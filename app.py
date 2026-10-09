
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import detect_anomalies
from lstm_model import prepare_time_series, train_lstm_autoencoder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VitalGuard AI",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 VitalGuard AI")
st.subheader("Intelligent Early-Warning System for Patient Vital Patterns")
st.caption("AI-powered • Data-driven • Explainable")
st.markdown("---")


# ============================================================
# WELCOME
# ============================================================

st.header("👋 Welcome to VitalGuard AI")

st.write(
    """
    VitalGuard AI is a research prototype designed to analyse
    ICU physiological time-series data and identify unusual
    patterns using machine-learning techniques.
    """
)

st.info(
    "⚕️ Research and educational prototype only. This app does not "
    "provide medical diagnosis or replace qualified healthcare professionals."
)


# ============================================================
# REFERENCE VITAL SIGNS
# ============================================================

st.markdown("---")
st.header("❤️ Reference Vital Signs")

st.write(
    "These general adult reference values are educational context only. "
    "They are not used as the AI anomaly-detection thresholds."
)

reference_data = pd.DataFrame(
    {
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
    }
)

st.dataframe(
    reference_data,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Reference values are general educational information, not individual medical advice."
)


# ============================================================
# PATIENT FILES
# ============================================================

patient_files = {
    "P001": "deploy_data/132539.txt",
    "P002": "deploy_data/132540.txt",
    "P003": "deploy_data/132541.txt",
}


# ============================================================
# PATIENT DATA INPUT
# ============================================================

st.markdown("---")
st.header("📤 Patient Data")

st.write(
    "Select a demonstration patient or upload a compatible ICU time-series file."
)

col1, col2 = st.columns(2)

with col1:
    selected_patient = st.selectbox(
        "Select Demonstration Patient",
        list(patient_files.keys()),
    )

with col2:
    uploaded_file = st.file_uploader(
        "Or upload patient data (.txt or .csv)",
        type=["txt", "csv"],
    )


# ============================================================
# SELECT DATA SOURCE
# ============================================================

uploaded_temp_path = None

if uploaded_file is not None:
    suffix = (
        ".csv"
        if uploaded_file.name.lower().endswith(".csv")
        else ".txt"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        uploaded_temp_path = temp_file.name

    file_path = uploaded_temp_path
    patient_display_name = "Uploaded Patient"

else:
    file_path = patient_files[selected_patient]
    patient_display_name = selected_patient


# ============================================================
# READ PATIENT DATA
# ============================================================

if not os.path.exists(file_path):
    st.error(
        f"Patient file not found: {file_path}. "
        "Check that the deploy_data folder and sample files are committed to GitHub."
    )
    st.stop()

try:
    raw_data = pd.read_csv(file_path)
except Exception as exc:
    st.error(f"Unable to read patient data: {exc}")
    st.stop()


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = ["Time", "Parameter", "Value"]

missing_columns = [
    column
    for column in required_columns
    if column not in raw_data.columns
]

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
        + ". Expected columns: Time, Parameter, Value."
    )
    st.stop()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown("---")
st.header("👤 Patient Information")

record_id_data = raw_data.loc[
    raw_data["Parameter"] == "RecordID",
    "Value",
]

record_id = (
    record_id_data.iloc[0]
    if not record_id_data.empty
    else "Unknown"
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Patient", patient_display_name)

with info2:
    st.metric("Record ID", str(record_id))

with info3:
    st.metric("Data Points", len(raw_data))

with st.expander("📋 View Raw Patient Data"):
    st.dataframe(
        raw_data,
        use_container_width=True,
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("---")

analyze_patient = st.button(
    "🔍 Analyze Patient",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_patient:

    # --------------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------------

    try:
        with st.spinner("Running Isolation Forest anomaly detection..."):
            time_series = detect_anomalies(file_path)

    except Exception as exc:
        st.error(f"Isolation Forest analysis failed: {exc}")
        st.stop()

    if time_series is None or time_series.empty:
        st.warning("No time-series observations were returned.")
        st.stop()

    if "Anomaly" not in time_series.columns:
        st.error(
            "The output from detect_anomalies() does not contain "
            "the required 'Anomaly' column."
        )
        st.stop()

    st.markdown("---")
    st.header("🌲 Isolation Forest Anomaly Detection")

    total_points = len(time_series)

    anomaly_data = time_series.loc[
        time_series["Anomaly"] == -1
    ]

    normal_data = time_series.loc[
        time_series["Anomaly"] == 1
    ]

    anomaly_count = len(anomaly_data)
    normal_count = len(normal_data)

    anomaly_rate = (
        anomaly_count / total_points * 100
        if total_points > 0
        else 0.0
    )

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric("Total Observations", total_points)

    with result2:
        st.metric("Detected Anomalies", anomaly_count)

    with result3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

    if anomaly_count == 0:
        st.success("🟢 No unusual patterns were flagged by this model.")
    elif anomaly_rate < 10:
        st.warning("🟡 A small number of unusual patterns were detected.")
    else:
        st.warning(
            "🟠 Multiple unusual patterns were detected and should be reviewed."
        )

    if anomaly_count > 0:
        st.subheader("⏱️ Detected Anomaly Periods")
        st.dataframe(
            anomaly_data,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PERSONALIZED BASELINE
    # --------------------------------------------------------

    st.markdown("---")
    st.header("📊 Personalized Patient Baseline")

    baseline = time_series.attrs.get(
        "baseline",
        pd.Series(dtype=float),
    )

    spread = time_series.attrs.get(
        "spread",
        pd.Series(dtype=float),
    )

    if isinstance(baseline, pd.Series) and not baseline.empty:

        baseline_table = pd.DataFrame(
            {
                "Vital Sign": baseline.index,
                "Patient Baseline": baseline.values,
                "Standard Deviation": [
                    spread.get(vital, np.nan)
                    if isinstance(spread, (pd.Series, dict))
                    else np.nan
                    for vital in baseline.index
                ],
            }
        )

        st.dataframe(
            baseline_table,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "This baseline is calculated from observed patient values. "
            "It is not a clinical reference range."
        )

    else:
        st.caption(
            "A saved patient baseline is not available from the current model output."
        )

    # --------------------------------------------------------
    # EXPLAINABLE ANOMALIES
    # --------------------------------------------------------

    st.markdown("---")
    st.header("🔎 Explainable Anomaly Results")

    deviation_score = time_series.attrs.get(
        "deviation_score",
        pd.DataFrame(),
    )

    if (
        anomaly_count > 0
        and isinstance(deviation_score, pd.DataFrame)
        and not deviation_score.empty
    ):

        explanations = []

        for timestamp in anomaly_data.index:

            if timestamp in deviation_score.index:

                row = (
                    deviation_score.loc[timestamp]
                    .abs()
                    .sort_values(ascending=False)
                    .head(3)
                )

                explanation = ", ".join(
                    f"{vital} ({value:.2f}σ)"
                    for vital, value in row.items()
                )

                explanations.append(
                    {
                        "Time": timestamp,
                        "Largest observed deviations": explanation,
                    }
                )

        if explanations:
            st.dataframe(
                pd.DataFrame(explanations),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "These are statistical deviations from the calculated baseline, "
                "not medical diagnoses."
            )

    else:
        st.caption(
            "Detailed deviation explanations are not available from the current model output."
        )

    # --------------------------------------------------------
    # LSTM AUTOENCODER
    # --------------------------------------------------------

    st.markdown("---")
    st.header("🧠 LSTM Autoencoder Analysis")

    lstm_result = None
    lstm_threshold = None

    try:
        lstm_data = prepare_time_series(file_path)

        if lstm_data is not None and len(lstm_data) >= 3:

            with st.spinner("Training LSTM Autoencoder..."):
                (
                    lstm_result,
                    lstm_model,
                    lstm_scaler,
                    lstm_threshold,
                ) = train_lstm_autoencoder(
                    lstm_data,
                    epochs=30,
                )

            st.success("LSTM temporal-pattern analysis completed.")

            if (
                lstm_result is not None
                and "LSTM_Anomaly" in lstm_result.columns
            ):
                lstm_anomalies = lstm_result.loc[
                    lstm_result["LSTM_Anomaly"].astype(bool)
                ]
            else:
                lstm_anomalies = pd.DataFrame()

            lstm1, lstm2 = st.columns(2)

            with lstm1:
                st.metric("LSTM Anomalies", len(lstm_anomalies))

            with lstm2:
                st.metric("LSTM Threshold", f"{float(lstm_threshold):.4f}")

            if not lstm_anomalies.empty:
                st.subheader("⏱️ LSTM Detected Periods")

                columns_to_show = [
                    column
                    for column in ["LSTM_Error", "LSTM_Anomaly"]
                    if column in lstm_anomalies.columns
                ]

                st.dataframe(
                    lstm_anomalies[columns_to_show],
                    use_container_width=True,
                )

        else:
            st.warning(
                "Not enough time-series observations for LSTM analysis."
            )

    except Exception as exc:
        st.error(f"LSTM analysis failed: {exc}")

    # --------------------------------------------------------
    # HYBRID AI
    # --------------------------------------------------------

    st.markdown("---")
    st.header("🤖 Hybrid AI Analysis")

    hybrid = None

    if (
        lstm_result is not None
        and "LSTM_Anomaly" in lstm_result.columns
    ):

        hybrid = time_series.copy()

        hybrid["IsolationForest"] = hybrid["Anomaly"].eq(-1)
        hybrid["LSTM"] = False

        common_times = hybrid.index.intersection(
            lstm_result.index
        )

        hybrid.loc[common_times, "LSTM"] = (
            lstm_result.loc[
                common_times,
                "LSTM_Anomaly",
            ].astype(bool)
        )

        hybrid["Hybrid_Anomaly"] = (
            hybrid["IsolationForest"] | hybrid["LSTM"]
        )

        hybrid1, hybrid2, hybrid3 = st.columns(3)

        with hybrid1:
            st.metric(
                "Isolation Forest",
                int(hybrid["IsolationForest"].sum()),
            )

        with hybrid2:
            st.metric(
                "LSTM",
                int(hybrid["LSTM"].sum()),
            )

        with hybrid3:
            st.metric(
                "Hybrid AI",
                int(hybrid["Hybrid_Anomaly"].sum()),
            )

        st.info(
            "Hybrid AI flags observations identified by either Isolation Forest "
            "or the LSTM Autoencoder. This combination is experimental."
        )

    else:
        st.info("Hybrid analysis requires successful LSTM analysis.")

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.markdown("---")
    st.header("📊 Model Comparison")

    comparison_rows = [
        {
            "Model": "Isolation Forest",
            "Flagged Observations": anomaly_count,
        }
    ]

    if (
        lstm_result is not None
        and "LSTM_Anomaly" in lstm_result.columns
    ):
        comparison_rows.append(
            {
                "Model": "LSTM Autoencoder",
                "Flagged Observations": int(
                    lstm_result["LSTM_Anomaly"].astype(bool).sum()
                ),
            }
        )

    if hybrid is not None:
        comparison_rows.append(
            {
                "Model": "Hybrid AI",
                "Flagged Observations": int(
                    hybrid["Hybrid_Anomaly"].sum()
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These are counts of observations flagged by each model, not precision, "
        "recall, or clinical performance metrics. Ground-truth labels are not "
        "available for this patient record."
    )

    if len(comparison_df) > 1:

        comparison_fig = go.Figure(
            data=[
                go.Bar(
                    x=comparison_df["Model"],
                    y=comparison_df["Flagged Observations"],
                    name="Flagged Observations",
                )
            ]
        )

        comparison_fig.update_layout(
            title="Number of Observations Flagged by Each Model",
            xaxis_title="Model",
            yaxis_title="Flagged Observations",
            height=400,
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # INTERACTIVE VITAL DASHBOARD
    # --------------------------------------------------------

    st.markdown("---")
    st.header("📈 Interactive Patient Vital Dashboard")

    possible_vitals = [
        "HR",
        "RespRate",
        "SaO2",
        "SpO2",
        "Temp",
        "SysABP",
        "DiasABP",
        "NISysABP",
        "NIDiasABP",
        "NIMAP",
    ]

    available_vitals = [
        vital
        for vital in possible_vitals
        if vital in time_series.columns
    ]

    if available_vitals:

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

        chart_data = (
            time_series[[selected_vital]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[selected_vital],
                mode="lines+markers",
                name=selected_vital,
                hovertemplate=(
                    f"<b>Time:</b> %{{x}}<br>"
                    f"<b>{selected_vital}:</b> %{{y:.2f}}"
                    "<extra></extra>"
                ),
            )
        )

        if show_anomalies:

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
                        marker={
                            "size": 12,
                            "symbol": "x",
                        },
                        hovertemplate=(
                            "<b>Anomaly</b><br>"
                            "<b>Time:</b> %{x}<br>"
                            f"<b>{selected_vital}:</b> %{{y:.2f}}"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title=f"{selected_vital} Timeline — {patient_display_name}",
            xaxis_title="Time",
            yaxis_title=selected_vital,
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("📊 Vital Sign Summary")

        summary1, summary2, summary3 = st.columns(3)

        with summary1:
            st.metric(
                "Average",
                f"{chart_data[selected_vital].mean():.2f}",
            )

        with summary2:
            st.metric(
                "Minimum",
                f"{chart_data[selected_vital].min():.2f}",
            )

        with summary3:
            st.metric(
                "Maximum",
                f"{chart_data[selected_vital].max():.2f}",
            )

    else:
        st.info(
            "No supported vital signs are available in the current time-series output."
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.markdown("---")
    st.header("🧠 AI Interpretation")

    if anomaly_count == 0:
        st.success(
            "The current Isolation Forest model did not flag unusual patterns "
            "in this record."
        )
    else:
        st.warning(
            f"The system flagged {anomaly_count} unusual observation(s) out of "
            f"{total_points} analysed observations."
        )

    st.write(
        "Statistical anomalies should be reviewed in context and are not medical diagnoses."
    )

    # --------------------------------------------------------
    # SDG 3
    # --------------------------------------------------------

    st.markdown("---")
    st.header("🌍 SDG 3 — Good Health and Well-being")

    st.write(
        "VitalGuard AI explores how data-driven AI may help highlight unusual "
        "physiological time-series patterns for further review, supporting "
        "research related to health and well-being."
    )

    st.info(
        "Research prototype only — this app does not replace doctors "
        "or provide medical diagnosis."
    )


# ============================================================
# INITIAL MESSAGE
# IMPORTANT: This else matches if analyze_patient above.
# ============================================================

else:
    st.info(
        "👆 Select P001, P002, or P003, or upload patient data, "
        "then click **🔍 Analyze Patient**."
    )


# ============================================================
# CLEAN UP TEMPORARY UPLOAD
# ============================================================

if (
    uploaded_temp_path is not None
    and os.path.exists(uploaded_temp_path)
):
    try:
        os.unlink(uploaded_temp_path)
    except OSError:
        pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VitalGuard AI • Python • Streamlit • Isolation Forest • "
    "LSTM Autoencoder • PhysioNet Challenge 2012 • SDG 3"
)
