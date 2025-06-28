import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
import pickle

# **Load and Preprocess Dataset**
df = pd.read_csv("./data/qr_analyzed_urls.csv")

# Drop rows with missing labels
df = df.dropna(subset=["threat_type"])

# **Updated Label Mapping (Binary Classification)**
label_mapping = {
    "malware": 1,  # Malicious
    "phishing": 1,  # Malicious
    "suspicious": 1,  # Malicious
    "safe": 0  # Safe
}

# Filter dataset to keep only "malicious" and "safe"
df = df[df["threat_type"].isin(label_mapping)]

# Convert text labels to binary labels
df["label"] = df["threat_type"].map(label_mapping)

# Extract URLs and labels
urls = df["URL"].astype(str).values
labels = df["label"].values

# **Convert text URLs to numerical features using TF-IDF**
vectorizer = TfidfVectorizer(max_features=1599)  
X = vectorizer.fit_transform(urls).toarray()

# **Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)  # Ensure correct dtype
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

# **Create DataLoader**
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# **Define CNN Model for Binary Classification**
class URLCNN(nn.Module):
    def __init__(self, input_size):
        super(URLCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)  # **Output changed to 2 classes (Safe/Malicious)**
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # No LogSoftmax (Handled by CrossEntropyLoss)

# **Initialize Model, Loss & Optimizer**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = URLCNN(input_size=1599).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **Train Model**
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# **Save TF-IDF Vectorizer**
with open("./models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# **Save Model**
torch.save(model.state_dict(), "./models/url_cnn_model.pt")
print(" Model retrained and saved successfully.")
