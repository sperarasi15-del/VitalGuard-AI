import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Patient file
file_path = "data/set-a/set-a/132539.txt"

# Read patient data
data = pd.read_csv(file_path)

# Vital signs we want to monitor
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

# Keep only selected vital signs
vitals = data[data["Parameter"].isin(vital_signs)].copy()

print("Selected vital signs:")
print(vitals)

print("\nNumber of vital-sign measurements:", len(vitals))

print("\nVital signs found:")
print(vitals["Parameter"].unique())

vitals = data[data["Parameter"].isin(vital_signs)].copy()

# Replace -1 with missing value
vitals["Value"] = vitals["Value"].replace(-1, pd.NA)

print("\nMissing values:")
print(vitals.isna().sum())


# Convert long format to time-series format
time_series = vitals.pivot(
    index="Time",
    columns="Parameter",
    values="Value"
)

print("\nTime-series data:")
print(time_series.head(10))

print("\nShape of time-series data:", time_series.shape)

# Fill missing values using the previous available measurement
time_series = time_series.ffill()

print("\nTime-series after filling missing values:")
print(time_series.head(10))

print("\nRemaining missing values:")
print(time_series.isna().sum())

# Prepare data for the anomaly detection model
# Standardize the vital signs
scaler = StandardScaler()
X = scaler.fit_transform(time_series)

print("\nStandardized data:")
print(X[:5])

print("\nData prepared for AI model:")
print(X.shape)

# Create the Isolation Forest model
model = IsolationForest(
    contamination=0.1,
    random_state=42
)

print("\nIsolation Forest model created successfully!")

# Train the model and predict anomalies
predictions = model.fit_predict(X)

print("\nAnomaly predictions:")
print(predictions)

# Add anomaly results to the time-series data
time_series["Anomaly"] = predictions

# Count detected anomalies
anomaly_count = (predictions == -1).sum()

print("\nNumber of anomalies detected:", anomaly_count)

print("\nAnomaly time points:")
print(time_series[time_series["Anomaly"] == -1])

# Plot Heart Rate and detected anomalies

plt.figure(figsize=(10, 5))

# Plot normal heart-rate values
normal = time_series[time_series["Anomaly"] == 1]
plt.plot(normal.index, normal["HR"], marker="o", label="Normal")

# Plot anomaly points
anomalies = time_series[time_series["Anomaly"] == -1]
plt.scatter(
    anomalies.index,
    anomalies["HR"],
    marker="x",
    s=100,
    label="Anomaly"
)

plt.xlabel("Time")
plt.ylabel("Heart Rate (bpm)")
plt.title("Heart Rate Anomaly Detection")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()