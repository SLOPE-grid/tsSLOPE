import os
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# --- Setup ---
# =====================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# --- Hyperparameters ---
# =====================================================
batch_size = 32
learning_rate = 1e-3
epochs = 300
PATHcwd = os.getcwd()

# =====================================================
# --- Load and Preprocess Data ---
# =====================================================
data_path = os.path.join(PATHcwd, "combined_llnl_data.mat")
data = sio.loadmat(data_path)["Data"]

# Binary target: last column >= 0 → class 1, else 0
TSI = data[:, -1].reshape(-1, 1)
TSI = (TSI >= 0).astype(int)

data = data[:, :-1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, TSI, test_size=0.2, random_state=42
)

# Reshape for 1D CNN: [batch, channels=1, features]
X_train_cnn = X_train.reshape(-1, 1, X_train.shape[1])
X_test_cnn = X_test.reshape(-1, 1, X_test.shape[1])

# Convert to tensors
X_train_t = torch.tensor(X_train_cnn, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_cnn, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)

# =====================================================
# --- Define Model ---
# =====================================================
class CNN1D_GELU_Avg(nn.Module):
    def __init__(self):
        super(CNN1D_GELU_Avg, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# --- Initialize Model, Loss, Optimizer ---
# =====================================================
model = CNN1D_GELU_Avg().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =====================================================
# --- Training ---
# =====================================================
print("\nStarting training...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.8f}")

run_time = round(time.time() - start_time, 3)

# =====================================================
# --- Evaluation ---
# =====================================================
model.eval()
with torch.no_grad():
    preds = model(X_test_t.to(device))
    y_pred = (preds >= 0.5).int().flatten().cpu().numpy()
    y_true = y_test_t.int().flatten().cpu().numpy()

accuracy_rate = (y_pred == y_true).mean()
RMSE = np.sqrt(mean_squared_error(y_true, preds.cpu().numpy()))
MAE = mean_absolute_error(y_true, preds.cpu().numpy())

print("\n--- Evaluation ---")
print(f"Accuracy: {accuracy_rate:.4f}")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE: {MAE:.6f}")

# =====================================================
# --- Save Model Parameters (Descriptive Filename) ---
# =====================================================
PATHmodel = (
    PATHcwd + r'/model_state_CNN1D_'
    + f'acc_{accuracy_rate:.5f}_'
    + f'rmse_{RMSE:.6f}_'
    + f'mae_{MAE:.6f}_'
    + f'epoch_{epochs}_'
    + f'bs_{batch_size}_'
    + f'lr_{learning_rate}_'
    + f'time_{run_time:.3f}.pth'
)

torch.save(model.state_dict(), PATHmodel)
print(f"\nModel parameters saved to:\n{PATHmodel}")
