import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
dataset_path = "./data/apk_dataset.csv"
df = pd.read_csv(dataset_path)

# Check for missing values
df.fillna("", inplace=True)  # Replace NaN with empty strings to avoid errors

# Ensure Dex_Size, Feature_Usage, and Network_Usage are numeric
numeric_columns = ["Dex_Size", "Feature_Usage", "Network_Usage"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, replace errors with NaN
df.fillna(0, inplace=True)  # Replace NaN values with 0

# Normalize numeric features
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Convert categorical features to lists
categorical_columns = ["Permissions", "API_Calls", "Activities", "Services", "Receivers", "Providers"]
for col in categorical_columns:
    df[col] = df[col].apply(lambda x: x.split(",") if isinstance(x, str) else [])

# Multi-Hot Encoding for categorical features
mlb = MultiLabelBinarizer()

encoded_features = []
for col in categorical_columns:
    encoded = mlb.fit_transform(df[col])  # Convert multi-label categorical data to binary matrix
    encoded_features.append(encoded)

# Combine all features into a single array
X = np.hstack((*encoded_features, df[numeric_columns].values))
y = df["Malware"].values  # Target: 0 (Benign), 1 (Malicious)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save processed data
processed_data = pd.DataFrame(X_resampled)
processed_data["label"] = y_resampled
processed_data.to_csv("./data/processed_apk_data.csv", index=False)

print(" Feature extraction completed. Data saved to processed_apk_data.csv")
