import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle  # <-- Add this to save training history
import matplotlib.pyplot as plt  # <-- Add this for visualization
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processed dataset
df = pd.read_csv("./data/processed_apk_data.csv")

# Extract Features & Labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors and Move to Device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN Model
class APK_CNN(nn.Module):
    def __init__(self, input_size):
        super(APK_CNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)  # No softmax (handled by CrossEntropyLoss)
        return x

# Initialize Model
input_size = X_train.shape[1]
model = APK_CNN(input_size).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# Lists to Store Training History
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate Training Accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # Validation Step
    model.eval()
    val_loss = 0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(val_correct / val_total)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]*100:.2f}%, Val Accuracy: {val_accuracies[-1]*100:.2f}%")

    scheduler.step()

# Save Model
torch.save(model.state_dict(), "./models/apk_cnn_model.pt")
print("Model training complete and saved!")

# Save Training History
history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}

with open("./models/training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print("Training history saved!")
