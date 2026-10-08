import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense


def prepare_time_series(file_path):

    data = pd.read_csv(file_path)

    vital_signs = [
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

    vitals = data[
        data["Parameter"].isin(vital_signs)
    ].copy()

    vitals["Value"] = vitals["Value"].replace(
        -1,
        np.nan
    )

    vitals["Value"] = pd.to_numeric(
        vitals["Value"],
        errors="coerce"
    )

    time_series = vitals.pivot(
        index="Time",
        columns="Parameter",
        values="Value"
    )

    time_series = time_series.ffill()
    time_series = time_series.dropna()

    return time_series


def train_lstm_autoencoder(
    time_series,
    epochs=30
):

    values = time_series.values

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        values
    )

    if len(scaled_values) < 3:

        raise ValueError(
            "Not enough time-series data for LSTM analysis."
        )

    sequence_length = 3

    X = []

    for i in range(
        len(scaled_values) - sequence_length + 1
    ):

        X.append(
            scaled_values[
                i:i + sequence_length
            ]
        )

    X = np.array(X)

    timesteps = X.shape[1]
    features = X.shape[2]

    inputs = Input(
        shape=(timesteps, features)
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

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        X,
        X,
        epochs=epochs,
        batch_size=8,
        verbose=0
    )

    reconstructed = model.predict(
        X,
        verbose=0
    )

    errors = np.mean(
        np.square(
            X - reconstructed
        ),
        axis=(1, 2)
    )

    threshold = np.percentile(
        errors,
        90
    )

    result = time_series.iloc[
        sequence_length - 1:
    ].copy()

    result["LSTM_Error"] = errors

    result["LSTM_Anomaly"] = (
        result["LSTM_Error"] > threshold
    )

    return result, model, scaler, threshold
