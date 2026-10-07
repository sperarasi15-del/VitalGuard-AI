import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(file_path):

    # Read patient data
    data = pd.read_csv(file_path)

    # Vital signs
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

    # Select vital signs
    vitals = data[data["Parameter"].isin(vital_signs)].copy()

    # Replace missing values
    vitals["Value"] = vitals["Value"].replace(-1, pd.NA)

    # Convert to time-series format
    time_series = vitals.pivot(
        index="Time",
        columns="Parameter",
        values="Value"
    )

    # Fill missing values
    time_series = time_series.ffill()

    # Remove rows that are still missing
    time_series = time_series.dropna()

    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(time_series)

    # Isolation Forest
    model = IsolationForest(
        contamination=0.1,
        random_state=42
    )

    predictions = model.fit_predict(X)

    # Store predictions
    time_series["Anomaly"] = predictions

    return time_series