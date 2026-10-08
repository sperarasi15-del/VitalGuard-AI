import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.graph_objects as go
import tensorflow as tf

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense
)

from model import detect_anomalies

from lstm_model import (
    prepare_time_series,
    train_lstm_autoencoder
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VitalGuard AI",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown("# 🏥 VitalGuard AI")

st.subheader(
    "Intelligent Early-Warning System for Patient Vital Patterns"
)

st.caption(
    "AI-powered • Data-driven • Explainable"
)

st.markdown("---")

# ============================================================
# WELCOME
# ============================================================

st.header("👋 Welcome to VitalGuard AI")

st.write(
    """
    VitalGuard AI is a research prototype designed to
    analyse ICU physiological time-series data and identify
    unusual patterns using machine-learning techniques.

    The system combines statistical anomaly detection,
    temporal-pattern analysis, and a hybrid AI approach.
    """
)

st.info(
    "⚕️ VitalGuard AI is a research and educational prototype. "
    "It does not provide medical diagnosis and does not replace "
    "qualified healthcare professionals."
)


# ============================================================
# REFERENCE VITAL SIGNS
# ============================================================

st.markdown("---")

st.header("❤️ Reference Vital Signs")

st.write(
    """
    These general adult reference values are shown only
    for educational context. They are NOT used as the
    AI anomaly-detection thresholds.
    """
)

reference_data = pd.DataFrame(
    {
        "Vital Sign": [
            "Heart Rate (HR)",
            "Respiratory Rate (RR)",
            "SpO₂",
            "Temperature",
            "Blood Pressure"
        ],

        "General Adult Reference": [
            "60–100 bpm",
            "12–20 breaths/min",
            "95–100%",
            "Approximately 36.1–37.2 °C",
            "Approximately 90/60–120/80 mmHg"
        ]
    }
)

st.dataframe(
    reference_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Reference values are general educational information "
    "and are not individual medical advice."
)


# ============================================================
# DEMONSTRATION PATIENT FILES
# ============================================================

patient_files = {
    "P001": "deploy_data/132539.txt",
    "P002": "deploy_data/132540.txt",
    "P003": "deploy_data/132541.txt"
}


# ============================================================
# PATIENT DATA
# ============================================================

st.markdown("---")

st.header("📤 Patient Data")

st.write(
    """
    Select a demonstration patient or upload your own
    compatible ICU time-series file.
    """
)


col1, col2 = st.columns(2)


with col1:

    selected_patient = st.selectbox(
        "Select Demonstration Patient",
        ["P001", "P002", "P003"]
    )


with col2:

    uploaded_file = st.file_uploader(
        "Or upload patient data (.txt or .csv)",
        type=["txt", "csv"]
    )


# ============================================================
# SELECT DATA SOURCE
# ============================================================

uploaded_temp_file = None


if uploaded_file is not None:

    uploaded_temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt"
    )

    uploaded_temp_file.write(
        uploaded_file.getvalue()
    )

    uploaded_temp_file.close()

    file_path = uploaded_temp_file.name

    patient_display_name = "Uploaded Patient"

else:

    file_path = patient_files[selected_patient]

    patient_display_name = selected_patient


# ============================================================
# READ PATIENT DATA
# ============================================================

try:

    raw_data = pd.read_csv(
        file_path
    )

except Exception as e:

    st.error(
        f"Unable to read patient data: {e}"
    )

    st.stop()


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "Time",
    "Parameter",
    "Value"
]


missing_columns = [
    column
    for column in required_columns
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


record_id_data = raw_data[
    raw_data["Parameter"] == "RecordID"
]


if len(record_id_data) > 0:

    record_id = record_id_data[
        "Value"
    ].iloc[0]

else:

    record_id = "Unknown"


info1, info2, info3 = st.columns(3)


with info1:

    st.metric(
        "Patient",
        patient_display_name
    )


with info2:

    st.metric(
        "Record ID",
        record_id
    )


with info3:

    st.metric(
        "Data Points",
        len(raw_data)
    )


with st.expander(
    "📋 View Raw Patient Data"
):

    st.dataframe(
        raw_data,
        use_container_width=True
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("---")

analyze_patient = st.button(
    "🔍 Analyze Patient",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_patient:

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    with st.spinner(
        "Running Isolation Forest anomaly detection..."
    ):

        try:

            time_series = detect_anomalies(
                file_path
            )

        except Exception as e:

            st.error(
                f"Isolation Forest analysis failed: {e}"
            )

            st.stop()


    # ========================================================
    # ISOLATION FOREST RESULTS
    # ========================================================

    st.markdown("---")

    st.header(
        "🌲 Isolation Forest Anomaly Detection"
    )


    total_points = len(
        time_series
    )


    anomaly_data = time_series[
        time_series["Anomaly"] == -1
    ]


    normal_data = time_series[
        time_series["Anomaly"] == 1
    ]


    anomaly_count = len(
        anomaly_data
    )


    normal_count = len(
        normal_data
    )


    anomaly_rate = (

        anomaly_count
        / total_points
        * 100

        if total_points > 0

        else 0
    )


    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "Total Observations",
            total_points
        )


    with result2:

        st.metric(
            "Detected Anomalies",
            anomaly_count
        )


    with result3:

        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.1f}%"
        )


    if anomaly_count == 0:

        st.success(
            "🟢 No unusual patterns detected."
        )

    elif anomaly_rate < 10:

        st.warning(
            "🟡 A small number of unusual patterns "
            "were detected."
        )

    else:

        st.warning(
            "🟠 Multiple unusual patterns were detected "
            "and should be reviewed."
        )


    if anomaly_count > 0:

        st.subheader(
            "⏱️ Detected Anomaly Periods"
        )

        st.dataframe(
            anomaly_data,
            use_container_width=True
        )


    # ========================================================
    # PERSONALIZED BASELINE
    # ========================================================

    st.markdown("---")

    st.header(
        "📊 Personalized Patient Baseline"
    )


    baseline = time_series.attrs.get(
        "baseline",
        pd.Series(dtype=float)
    )


    spread = time_series.attrs.get(
        "spread",
        pd.Series(dtype=float)
    )


    if len(baseline) > 0:

        baseline_table = pd.DataFrame(
            {
                "Vital Sign": baseline.index,

                "Patient Baseline":
                    baseline.values,

                "Standard Deviation": [
                    spread.get(
                        vital,
                        np.nan
                    )

                    for vital
                    in baseline.index
                ]
            }
        )


        st.dataframe(
            baseline_table,
            use_container_width=True,
            hide_index=True
        )


        st.info(
            "The baseline is calculated from the observed "
            "values for this patient. It is not a clinical "
            "reference range."
        )


    # ========================================================
    # EXPLAINABLE ANOMALIES
    # ========================================================

    st.markdown("---")

    st.header(
        "🔎 Explainable Anomaly Results"
    )


    deviation_score = time_series.attrs.get(
        "deviation_score",
        pd.DataFrame()
    )


    if (
        anomaly_count > 0
        and not deviation_score.empty
    ):

        explanations = []


        for timestamp in anomaly_data.index:

            if timestamp in deviation_score.index:

                row = (
                    deviation_score
                    .loc[timestamp]
                    .abs()
                    .sort_values(
                        ascending=False
                    )
                )


                top_vitals = row.head(3)


                explanation = ", ".join(
                    [
                        f"{vital} ({value:.2f}σ)"

                        for vital, value
                        in top_vitals.items()
                    ]
                )


                explanations.append(
                    {
                        "Time": timestamp,

                        "Largest observed deviations":
                            explanation
                    }
                )


        if explanations:

            explanation_df = pd.DataFrame(
                explanations
            )


            st.dataframe(
                explanation_df,
                use_container_width=True,
                hide_index=True
            )


            st.caption(
                "These vital signs showed the largest "
                "observed deviations from the patient's "
                "calculated baseline. They do not represent "
                "a medical diagnosis."
            )


    # ========================================================
    # LSTM AUTOENCODER
    # ========================================================

    st.markdown("---")

    st.header(
        "🧠 LSTM Autoencoder Analysis"
    )


    lstm_result = None


    try:

        lstm_data = prepare_time_series(
            file_path
        )


        if len(lstm_data) >= 3:

            with st.spinner(
                "Training LSTM Autoencoder..."
            ):

                (
                    lstm_result,
                    lstm_model,
                    lstm_scaler,
                    lstm_threshold
                ) = train_lstm_autoencoder(
                    lstm_data,
                    epochs=30
                )


            st.success(
                "LSTM temporal-pattern analysis completed."
            )


            lstm_anomalies = lstm_result[
                lstm_result[
                    "LSTM_Anomaly"
                ] == True
            ]


            lstm1, lstm2 = st.columns(2)


            with lstm1:

                st.metric(
                    "LSTM Anomalies",
                    len(lstm_anomalies)
                )


            with lstm2:

                st.metric(
                    "LSTM Threshold",
                    f"{lstm_threshold:.4f}"
                )


            if len(lstm_anomalies) > 0:

                st.subheader(
                    "⏱️ LSTM Detected Periods"
                )


                st.dataframe(
                    lstm_anomalies[
                        [
                            "LSTM_Error",
                            "LSTM_Anomaly"
                        ]
                    ],
                    use_container_width=True
                )


        else:

            st.warning(
                "Not enough time-series observations "
                "for LSTM analysis."
            )


    except Exception as e:

        st.error(
            f"LSTM analysis failed: {e}"
        )


    # ========================================================
    # HYBRID AI
    # ========================================================

    st.markdown("---")

    st.header(
        "🤖 Hybrid AI Analysis"
    )


    hybrid = None


    if lstm_result is not None:

        hybrid = time_series.copy()


        hybrid[
            "IsolationForest"
        ] = (
            hybrid["Anomaly"] == -1
        )


        common_times = (
            hybrid.index
            .intersection(
                lstm_result.index
            )
        )


        hybrid["LSTM"] = False


        hybrid.loc[
            common_times,
            "LSTM"
        ] = (
            lstm_result.loc[
                common_times,
                "LSTM_Anomaly"
            ]
            .astype(bool)
        )


        hybrid[
            "Hybrid_Anomaly"
        ] = (
            hybrid["IsolationForest"]
            | hybrid["LSTM"]
        )


        hybrid_count = int(
            hybrid[
                "Hybrid_Anomaly"
            ].sum()
        )


        hybrid1, hybrid2, hybrid3 = (
            st.columns(3)
        )


        with hybrid1:

            st.metric(
                "Isolation Forest",
                int(
                    hybrid[
                        "IsolationForest"
                    ].sum()
                )
            )


        with hybrid2:

            st.metric(
                "LSTM",
                int(
                    hybrid["LSTM"].sum()
                )
            )


        with hybrid3:

            st.metric(
                "Hybrid AI",
                hybrid_count
            )


        st.info(
            "Hybrid AI combines statistical anomaly "
            "detection from Isolation Forest with temporal "
            "pattern detection from the LSTM Autoencoder."
        )


    else:

        st.info(
            "Hybrid analysis requires successful "
            "LSTM analysis."
        )


    # ========================================================
    # CORRECTED MODEL COMPARISON
    # ========================================================

    st.markdown("---")

    st.header(
        "📊 Model Comparison"
    )


    st.subheader(
        f"{patient_display_name} — AI Model Performance"
    )


    st.write(
        """
        The models are evaluated using a controlled
        synthetic benchmark. The models are trained on
        clean patient observations and evaluated on a
        separate test copy containing known artificial
        anomalies.
        """
    )


    st.warning(
        "⚠️ Synthetic benchmark results are for algorithm "
        "comparison only and are NOT clinical ground truth."
    )


    try:

        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        clean_data = prepare_time_series(
            file_path
        )


        if len(clean_data) < 8:

            raise ValueError(
                "Not enough observations for model comparison."
            )


        clean_data = (
            clean_data
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .ffill()
            .bfill()
            .dropna()
        )


        if len(clean_data) < 8:

            raise ValueError(
                "Not enough valid observations "
                "after preprocessing."
            )


        # ----------------------------------------------------
        # TEST DATA
        # ----------------------------------------------------

        benchmark_test = clean_data.copy()


        n = len(
            benchmark_test
        )


        # Three known anomaly positions
        anomaly_positions = [
            n // 4,
            n // 2,
            (3 * n) // 4
        ]


        anomaly_positions = sorted(
            set(
                position
                for position
                in anomaly_positions

                if 0 <= position < n
            )
        )


        anomaly_times = [
            benchmark_test.index[position]

            for position
            in anomaly_positions
        ]


        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        ground_truth = np.zeros(
            n,
            dtype=bool
        )


        for position in anomaly_positions:

            ground_truth[position] = True


        # ----------------------------------------------------
        # INSERT SYNTHETIC ANOMALIES
        # ----------------------------------------------------

        if "HR" in benchmark_test.columns:

            target_vital = "HR"

        else:

            target_vital = (
                benchmark_test.columns[0]
            )


        vital_std = benchmark_test[
            target_vital
        ].std()


        if (
            pd.isna(vital_std)
            or vital_std == 0
        ):

            vital_std = 1.0


        for position in anomaly_positions:

            original_value = (
                benchmark_test.iloc[
                    position
                ][target_vital]
            )


            new_value = (
                original_value
                + 6 * vital_std
            )


            benchmark_test.iloc[
                position,
                benchmark_test.columns.get_loc(
                    target_vital
                )
            ] = new_value


        # ----------------------------------------------------
        # ISOLATION FOREST
        # ----------------------------------------------------

        isolation_scaler = (
            StandardScaler()
        )


        clean_scaled = (
            isolation_scaler.fit_transform(
                clean_data
            )
        )


        test_scaled = (
            isolation_scaler.transform(
                benchmark_test
            )
        )


        isolation_model = IsolationForest(
            contamination=0.1,
            random_state=42
        )


        # Train on CLEAN data
        isolation_model.fit(
            clean_scaled
        )


        # Predict TEST data
        isolation_predictions = (
            isolation_model.predict(
                test_scaled
            ) == -1
        )


        # ----------------------------------------------------
        # LSTM AUTOENCODER
        # ----------------------------------------------------

        np.random.seed(42)

        tf.random.set_seed(42)


        lstm_scaler = (
            StandardScaler()
        )


        clean_scaled_lstm = (
            lstm_scaler
            .fit_transform(
                clean_data.values
            )
        )


        test_scaled_lstm = (
            lstm_scaler
            .transform(
                benchmark_test.values
            )
        )


        sequence_length = 3


        # ----------------------------------------------------
        # CLEAN TRAINING SEQUENCES
        # ----------------------------------------------------

        clean_sequences = []


        for i in range(
            len(clean_scaled_lstm)
            - sequence_length
            + 1
        ):

            clean_sequences.append(
                clean_scaled_lstm[
                    i:
                    i + sequence_length
                ]
            )


        clean_sequences = np.array(
            clean_sequences
        )


        # ----------------------------------------------------
        # TEST SEQUENCES
        # ----------------------------------------------------

        test_sequences = []


        for i in range(
            len(test_scaled_lstm)
            - sequence_length
            + 1
        ):

            test_sequences.append(
                test_scaled_lstm[
                    i:
                    i + sequence_length
                ]
            )


        test_sequences = np.array(
            test_sequences
        )


        timesteps = (
            clean_sequences.shape[1]
        )


        features = (
            clean_sequences.shape[2]
        )


        # ----------------------------------------------------
        # BUILD LSTM
        # ----------------------------------------------------

        inputs = Input(
            shape=(
                timesteps,
                features
            )
        )


        encoded = LSTM(
            32,
            activation="tanh"
        )(inputs)


        repeated = RepeatVector(
            timesteps
        )(encoded)


        decoded = LSTM(
            32,
            activation="tanh",
            return_sequences=True
        )(repeated)


        outputs = TimeDistributed(
            Dense(features)
        )(decoded)


        benchmark_lstm_model = Model(
            inputs,
            outputs
        )


        benchmark_lstm_model.compile(
            optimizer="adam",
            loss="mse"
        )


        # ----------------------------------------------------
        # TRAIN LSTM ON CLEAN DATA
        # ----------------------------------------------------

        with st.spinner(
            "Training LSTM for model comparison..."
        ):

            benchmark_lstm_model.fit(
                clean_sequences,
                clean_sequences,
                epochs=30,
                batch_size=8,
                verbose=0
            )


        # ----------------------------------------------------
        # CLEAN RECONSTRUCTION ERROR
        # ----------------------------------------------------

        clean_reconstructed = (
            benchmark_lstm_model
            .predict(
                clean_sequences,
                verbose=0
            )
        )


        clean_errors = np.mean(
            np.square(
                clean_sequences
                - clean_reconstructed
            ),
            axis=(1, 2)
        )


        lstm_threshold = np.percentile(
            clean_errors,
            90
        )


        # ----------------------------------------------------
        # TEST RECONSTRUCTION ERROR
        # ----------------------------------------------------

        test_reconstructed = (
            benchmark_lstm_model
            .predict(
                test_sequences,
                verbose=0
            )
        )


        test_errors = np.mean(
            np.square(
                test_sequences
                - test_reconstructed
            ),
            axis=(1, 2)
        )


        # ----------------------------------------------------
        # LSTM PREDICTIONS
        # ----------------------------------------------------

        lstm_predictions = np.zeros(
            n,
            dtype=bool
        )


        sequence_predictions = (
            test_errors
            > lstm_threshold
        )


        for i, prediction in enumerate(
            sequence_predictions
        ):

            if prediction:

                time_position = (
                    i
                    + sequence_length
                    - 1
                )


                if time_position < n:

                    lstm_predictions[
                        time_position
                    ] = True


        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        hybrid_predictions = (
            isolation_predictions
            | lstm_predictions
        )


        # ----------------------------------------------------
        # METRICS FUNCTION
        # ----------------------------------------------------

        def calculate_metrics(
            ground_truth_values,
            predictions
        ):

            precision = precision_score(
                ground_truth_values,
                predictions,
                zero_division=0
            )


            recall = recall_score(
                ground_truth_values,
                predictions,
                zero_division=0
            )


            f1 = f1_score(
                ground_truth_values,
                predictions,
                zero_division=0
            )


            confusion = confusion_matrix(
                ground_truth_values,
                predictions,
                labels=[
                    False,
                    True
                ]
            )


            tn, fp, fn, tp = (
                confusion.ravel()
            )


            false_positive_rate = (

                fp / (fp + tn)

                if (fp + tn) > 0

                else 0
            )


            return (
                precision,
                recall,
                f1,
                false_positive_rate
            )


        # ----------------------------------------------------
        # CALCULATE ALL MODEL METRICS
        # ----------------------------------------------------

        isolation_metrics = (
            calculate_metrics(
                ground_truth,
                isolation_predictions
            )
        )


        lstm_metrics = (
            calculate_metrics(
                ground_truth,
                lstm_predictions
            )
        )


        hybrid_metrics = (
            calculate_metrics(
                ground_truth,
                hybrid_predictions
            )
        )


        # ----------------------------------------------------
        # COMPARISON DATAFRAME
        # ----------------------------------------------------

        evaluation_df = pd.DataFrame(
            {
                "Model": [
                    "Isolation Forest",
                    "LSTM Autoencoder",
                    "Hybrid AI"
                ],

                "Precision": [
                    isolation_metrics[0],
                    lstm_metrics[0],
                    hybrid_metrics[0]
                ],

                "Recall": [
                    isolation_metrics[1],
                    lstm_metrics[1],
                    hybrid_metrics[1]
                ],

                "F1-Score": [
                    isolation_metrics[2],
                    lstm_metrics[2],
                    hybrid_metrics[2]
                ],

                "Detection Rate": [
                    isolation_metrics[1],
                    lstm_metrics[1],
                    hybrid_metrics[1]
                ],

                "False Positive Rate": [
                    isolation_metrics[3],
                    lstm_metrics[3],
                    hybrid_metrics[3]
                ]
            }
        )


        display_df = (
            evaluation_df.copy()
        )


        metric_columns = [
            "Precision",
            "Recall",
            "F1-Score",
            "Detection Rate",
            "False Positive Rate"
        ]


        display_df[
            metric_columns
        ] = (
            display_df[
                metric_columns
            ].round(3)
        )


        # ----------------------------------------------------
        # DISPLAY TABLE
        # ----------------------------------------------------

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model = (
            evaluation_df
            .sort_values(
                "F1-Score",
                ascending=False
            )
            .iloc[0]
        )


        st.success(
            f"🏆 Best F1-Score: "
            f"{best_model['Model']} "
            f"({best_model['F1-Score']:.3f})"
        )


        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------

        st.subheader(
            "📊 AI Model Comparison Graph"
        )


        comparison_fig = (
            go.Figure()
        )


        comparison_fig.add_trace(
            go.Bar(
                x=evaluation_df[
                    "Model"
                ],

                y=evaluation_df[
                    "Precision"
                ],

                name="Precision"
            )
        )


        comparison_fig.add_trace(
            go.Bar(
                x=evaluation_df[
                    "Model"
                ],

                y=evaluation_df[
                    "Recall"
                ],

                name="Recall"
            )
        )


        comparison_fig.add_trace(
            go.Bar(
                x=evaluation_df[
                    "Model"
                ],

                y=evaluation_df[
                    "F1-Score"
                ],

                name="F1-Score"
            )
        )


        comparison_fig.update_layout(
            title=(
                f"{patient_display_name} — "
                "AI Model Comparison"
            ),

            xaxis_title="AI Model",

            yaxis_title="Performance Score",

            yaxis=dict(
                range=[0, 1]
            ),

            barmode="group",

            height=500,

            legend_title="Metrics",

            margin=dict(
                l=50,
                r=50,
                t=80,
                b=80
            )
        )


        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
            key=(
                f"model_comparison_"
                f"{patient_display_name}"
            )
        )


        # ----------------------------------------------------
        # BENCHMARK INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "🧪 Synthetic Benchmark"
        )


        st.write(
            f"Target vital used for the synthetic "
            f"benchmark: **{target_vital}**"
        )


        st.write(
            "Known synthetic anomaly times:"
        )


        st.write(
            ", ".join(
                str(time)
                for time
                in anomaly_times
            )
        )


        st.write(
            "Synthetic anomaly magnitude: approximately "
            "6 standard deviations of the selected vital."
        )


        st.info(
            "The models were trained using clean patient "
            "observations and evaluated on a separate copy "
            "containing controlled synthetic anomalies. "
            "These results are for algorithm comparison "
            "and are not clinical validation."
        )


    except Exception as e:

        st.error(
            f"Model comparison failed: {e}"
        )


    # ========================================================
    # INTERACTIVE VITAL DASHBOARD
    # ========================================================

    st.markdown("---")

    st.header(
        "📈 Interactive Patient Vital Dashboard"
    )


    available_vitals = [
        vital

        for vital in [
            "HR",
            "RespRate",
            "SaO2",
            "Temp",
            "SysABP",
            "DiasABP",
            "NISysABP",
            "NIDiasABP",
            "NIMAP"
        ]

        if vital in time_series.columns
    ]


    if len(available_vitals) > 0:

        chart_col1, chart_col2 = (
            st.columns(2)
        )


        with chart_col1:

            selected_vital = st.selectbox(
                "Select Vital Sign",
                available_vitals
            )


        with chart_col2:

            show_anomalies = st.checkbox(
                "Show Isolation Forest anomalies",
                value=True
            )


        chart_data = (
            time_series[
                [selected_vital]
            ]
            .dropna()
        )


        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=chart_data.index,

                y=chart_data[
                    selected_vital
                ],

                mode="lines+markers",

                name=selected_vital,

                hovertemplate=(
                    "<b>Time:</b> %{x}<br>"

                    f"<b>{selected_vital}:</b> "

                    "%{y:.2f}"

                    "<extra></extra>"
                )
            )
        )


        if show_anomalies:

            anomaly_times = (
                time_series[
                    time_series[
                        "Anomaly"
                    ] == -1
                ].index
            )


            anomaly_points = (
                chart_data[
                    chart_data.index.isin(
                        anomaly_times
                    )
                ]
            )


            if len(
                anomaly_points
            ) > 0:

                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points.index,

                        y=anomaly_points[
                            selected_vital
                        ],

                        mode="markers",

                        name="Anomaly",

                        marker=dict(
                            size=12,
                            symbol="x"
                        ),

                        hovertemplate=(
                            "<b>Anomaly</b><br>"

                            "<b>Time:</b> %{x}<br>"

                            f"<b>{selected_vital}:</b> "

                            "%{y:.2f}"

                            "<extra></extra>"
                        )
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

            height=500
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            key=(
                f"vital_chart_"
                f"{patient_display_name}_"
                f"{selected_vital}"
            )
        )


        # ----------------------------------------------------
        # VITAL SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📊 Vital Sign Summary"
        )


        summary1, summary2, summary3 = (
            st.columns(3)
        )


        with summary1:

            st.metric(
                "Average",
                f"{chart_data[selected_vital].mean():.2f}"
            )


        with summary2:

            st.metric(
                "Minimum",
                f"{chart_data[selected_vital].min():.2f}"
            )


        with summary3:

            st.metric(
                "Maximum",
                f"{chart_data[selected_vital].max():.2f}"
            )


    else:

        st.info(
            "No supported vital signs are available."
        )


    # ========================================================
    # AI INTERPRETATION
    # ========================================================

    st.markdown("---")

    st.header(
        "🧠 AI Interpretation"
    )


    if anomaly_count == 0:

        st.success(
            "The analyzed patient data did not show "
            "unusual patterns according to the current "
            "Isolation Forest model."
        )

    else:

        st.warning(
            f"The system detected {anomaly_count} "
            f"unusual observation(s) out of "
            f"{total_points} analyzed time points."
        )


        st.write(
            "The detected patterns represent statistical "
            "anomalies in the observed physiological data. "
            "They should be reviewed in context and are "
            "not medical diagnoses."
        )


    # ========================================================
    # SDG 3
    # ========================================================

    st.markdown("---")

    st.header(
        "🌍 SDG 3 — Good Health and Well-being"
    )


    st.write(
        """
        VitalGuard AI supports SDG 3 by exploring how
        data-driven artificial intelligence can assist
        with monitoring physiological time-series data
        and highlighting unusual patterns for further review.
        """
    )


    st.info(
        "Research prototype only — VitalGuard AI does not "
        "replace doctors or provide medical diagnosis."
    )


    # ========================================================
    # DELETE TEMPORARY UPLOAD
    # ========================================================

    if uploaded_temp_file is not None:

        try:

            os.unlink(
                file_path
            )

        except Exception:

            pass


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👆 Select P001, P002, or P003, or upload patient "
        "data, then click **🔍 Analyze Patient**."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VitalGuard AI • Python • Streamlit • "
    "Isolation Forest • LSTM Autoencoder • "
    "PhysioNet Challenge 2012 • SDG 3"
)