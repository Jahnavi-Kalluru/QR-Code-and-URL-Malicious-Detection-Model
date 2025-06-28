import pickle
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from train_apk_cnn import APK_CNN  # Ensure this is correctly imported
from sklearn.metrics import classification_report, accuracy_score

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processed dataset
df = pd.read_csv("./data/processed_apk_data.csv")

# Extract Features & Labels (Ensure consistency with training)
X = df.drop(columns=["label"]).values
y = df["label"].values
input_size = X.shape[1]  # Dynamically get correct feature size

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)

# Create Test DataLoader (No shuffle needed)
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize Model with Correct Input Size
model = APK_CNN(input_size).to(device)
model.load_state_dict(torch.load("./models/apk_cnn_model.pt"))
model.eval()

# Load Training History
with open("./models/training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history["train_losses"], label="Training Loss")
plt.plot(history["val_losses"], label="Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Plot Training & Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history["train_accuracies"], label="Training Accuracy")
plt.plot(history["val_accuracies"], label="Validation Accuracy", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# Compute Confusion Matrix
y_true, y_pred = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1)
        y_true.extend(batch_y.tolist())
        y_pred.extend(predictions.tolist())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

accuracy = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print Classification Report
class_names = ["Benign", "Malicious"]
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
