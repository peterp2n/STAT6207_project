# modeling_basic_reverse_with_val.py

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os

# -------------------------------
# Configuration & Paths
# -------------------------------
os.chdir('/Users/ty/Downloads/STAT6207/STAT6207_project')  # Adjust if needed

torch.manual_seed(42)

data_folder = Path("data")
data_folder.mkdir(parents=True, exist_ok=True)

train_path = data_folder / "train_all_cols_unstd_v2.csv"
test_path  = data_folder / "test_all_cols_unstd_v2.csv"

# -------------------------------
# Load Data
# -------------------------------
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

TARGET_COL     = "Next_Q1"
OPP_TARGET_COL = "Next_Q1_log1p"

y_train_full = train_df[OPP_TARGET_COL]
y_test       = test_df[OPP_TARGET_COL]

cols_to_drop = ['isbn', TARGET_COL, 'Next_Q2', 'Next_Q3', 'Next_Q4']

X_train_full = train_df.drop(columns=cols_to_drop)
X_test       = test_df.drop(columns=cols_to_drop)

# Ensure identical column order
X_test = X_test[X_train_full.columns]

# -------------------------------
# Train / Validation Split (80/20)
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_tensor   = torch.tensor(X_val.values,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1, 1)

X_test_tensor  = torch.tensor(X_test.values,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

# Move validation & test tensors to device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_val_device  = X_val_tensor.to(device)
y_val_device  = y_val_tensor.to(device)
X_test_device = X_test_tensor.to(device)

# -------------------------------
# DataLoader
# -------------------------------
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# Model Definition
# -------------------------------
class Regressor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train_tensor.shape[1]
model = Regressor(input_dim=input_dim, dropout=0.0).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# -------------------------------
# Training Loop with Validation
# -------------------------------
epochs = 100
loss_history      = []   # Train MSE per epoch
val_loss_history  = []   # Validation MSE per epoch

print(f"{'Epoch':<8} {'Train MSE':<15} {'Train RMSE':<15} {'Val MSE':<15} {'Val RMSE':<15}")
print("-" * 75)

for epoch in range(epochs):
    # ---------- Training ----------
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    train_mse  = epoch_loss / len(train_loader.dataset)
    train_rmse = np.sqrt(train_mse)
    loss_history.append(train_mse)

    # ---------- Validation ----------
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_device)
        val_loss  = criterion(val_preds, y_val_device)
        val_mse   = val_loss.item()
        val_rmse  = np.sqrt(val_mse)
        val_loss_history.append(val_mse)

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch + 1:<8} {train_mse:<15.6f} {train_rmse:<15.6f} {val_mse:<15.6f} {val_rmse:<15.6f}")

print("-" * 95)

# -------------------------------
# Final Test Evaluation
# -------------------------------
model.eval()
with torch.no_grad():
    preds_test = model(X_test_device).cpu().numpy().flatten()

true_test = y_test_tensor.numpy().flatten()

test_mse  = mean_squared_error(true_test, preds_test)
test_rmse = np.sqrt(test_mse)

print(f"\nFinal Test Results:")
print(f"Test MSE:  {test_mse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")

# -------------------------------
# Results DataFrame (optional)
# -------------------------------
results_df = pd.DataFrame({
    "true_next_q1_log1p": true_test,
    "pred_next_q1_log1p": preds_test
})
print("\nFirst few predictions:")
print(results_df.head())

# -------------------------------
# Plotting
# -------------------------------
# 1. Training vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history,     label="Train MSE", linewidth=2)
plt.plot(val_loss_history, label="Validation MSE", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.ylim(bottom=0, top=5)
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. True vs Predicted on Test Set
plt.figure(figsize=(7, 7))
plt.scatter(true_test, preds_test, alpha=0.6, edgecolor='k', linewidth=0.5)
min_val = min(true_test.min(), preds_test.min())
max_val = max(true_test.max(), preds_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
plt.xlabel("True Next_Q1_log1p")
plt.ylabel("Predicted Next_Q1_log1p")
plt.title(f"True vs Predicted (Test RMSE = {test_rmse:.4f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Training and evaluation completed.")