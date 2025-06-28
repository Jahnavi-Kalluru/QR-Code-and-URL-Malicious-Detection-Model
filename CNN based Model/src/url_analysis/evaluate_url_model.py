import torch
import torch.nn as nn
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# **Load Label Mapping**
label_mapping = {
    "malware": 1,
    "phishing": 1,
    "suspicious": 1,
    "safe": 0
}
reverse_mapping = {v: k for k, v in label_mapping.items()}  

# **Load Model & Define CNN**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class URLCNN(nn.Module):
    def __init__(self, input_size):
        super(URLCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)  # **Binary Classification**
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # No LogSoftmax (Handled by CrossEntropyLoss)

# **Load Model Weights**
model = URLCNN(input_size=1599).to(device)
model.load_state_dict(torch.load("./models/url_cnn_model.pt", map_location=device))
model.eval()

# **Load Dataset for Evaluation**
df = pd.read_csv("./data/qr_analyzed_urls.csv")
df = df[df["threat_type"].isin(label_mapping)]  # Filter unknown labels
df["label"] = df["threat_type"].map(label_mapping)

# **Load TF-IDF Vectorizer**
with open("./models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# **Transform URLs using TF-IDF**
urls = df["URL"].astype(str).values
X = vectorizer.transform(urls).toarray()
y_true = df["label"].values

# **Convert to Tensors**
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_true, dtype=torch.long).to(device)

# **Create DataLoader for Evaluation**
test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False)

# **Evaluate Model**
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        predictions = torch.argmax(outputs, dim=1)
        y_pred.extend(predictions.tolist())

# **Metrics Calculation**
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)

# **Print Evaluation Metrics**
print("\n **Model Evaluation Metrics**")
print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")

# **Plot Confusion Matrix**
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Safe", "Malicious"], yticklabels=["Safe", "Malicious"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# **Load Training History for Loss & Accuracy Visualization**
with open("./models/training_history.pkl", "rb") as f:
    history = pickle.load(f)

# **Plot Training vs Validation Loss**
plt.figure(figsize=(10, 5))
plt.plot(history["train_losses"], label="Training Loss", marker="o")
plt.plot(history["val_losses"], label="Validation Loss", linestyle="dashed", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# **Plot Training vs Validation Accuracy**
plt.figure(figsize=(10, 5))
plt.plot(history["train_accuracies"], label="Training Accuracy", marker="o")
plt.plot(history["val_accuracies"], label="Validation Accuracy", linestyle="dashed", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()
