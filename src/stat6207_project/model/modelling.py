import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

# Paths to the CSV files
data_folder = Path("data")
train_path = data_folder / "train_all_cols_unstd_v2.csv"
test_path = data_folder / "test_all_cols_unstd_v2.csv"


# Load the data
train_df = (
    pl.read_csv(train_path,
                schema_overrides={"isbn": pl.Utf8, "number_of_reviews": pl.Float64})
    .with_columns([pl.col("number_of_reviews").round(0).cast(pl.Int32).alias("number_of_reviews"),])
)
test_df = (
    pl.read_csv(test_path,
                schema_overrides={"isbn": pl.Utf8, "number_of_reviews": pl.Float64})
    .with_columns(pl.col("number_of_reviews").round(0).cast(pl.Int32).alias("number_of_reviews"))
)

# 1. Identify numeric columns
cont_var = train_df.select(cs.numeric()).columns
onehot_cols = [col for col in cont_var if "format" in col or "age" in col] + ["channel"]
cont_var = [col for col in cont_var if col not in onehot_cols]

# 2. Fit and transform (returns a 2D numpy array)
scaler = StandardScaler()

# 3. Assign back by mapping array columns to names
train_df = train_df.with_columns(
    pl.Series(name=c, values=scaler.fit_transform(train_df.select(cont_var))[:, i])
    for i, c in enumerate(cont_var)
)

test_df = test_df.with_columns(
    pl.Series(name=c, values=scaler.transform(test_df.select(cont_var))[:, i])
    for i, c in enumerate(cont_var)
)


# Separate features and target
TARGET_COL = "Next_Q1"
unwanted_cols = ["isbn", "Next_Q1_log1p", "Next_Q2", "Next_Q3", "Next_Q4", "customer_reviews",
                 "number_of_reviews"]

X_train = train_df.drop([TARGET_COL] + unwanted_cols)
y_train = train_df[TARGET_COL]

X_test = test_df.drop([TARGET_COL] + unwanted_cols)
y_test = test_df[TARGET_COL]

# Enforce same columns and order on test
X_test = X_test[X_train.columns]

# Convert to tensors
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

# Check for nulls in train and test dataframes
print("Null counts in train_df:")
print(train_df.null_count())
print("\nNull counts in test_df:")
print(test_df.null_count())

# Create DataLoaders
batch_size = 32
learning_rate = 0.00001
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define a simple feedforward neural network for regression
class Regressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


input_dim = X_train_tensor.shape[1]
model = Regressor(input_dim)

# Use GPU if available, with MPS support on Apple Silicon
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Running on: {device}")
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- PREPARE VALIDATION DATA ON DEVICE ONCE ---
# Moving this outside the loop saves overhead
X_val_device = X_test_tensor.to(device)
y_val_device = y_test_tensor.to(device)

# Training loop
epochs = 100
model.train()
loss_history = []
val_loss_history = []  # Optional: track val loss for plotting

print("\nTraining Progress:")
print("-" * 95)
# Updated header to include Val metrics
print(f"{'Epoch':<8} {'Train MSE':<15} {'Train RMSE':<15} {'Val MSE':<15} {'Val RMSE':<15}")
print("-" * 95)

for epoch in range(epochs):
    epoch_loss = 0.0

    # Training Step
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    # Calculate Train metrics
    epoch_mse = epoch_loss / len(train_loader.dataset)
    epoch_rmse = np.sqrt(epoch_mse)
    loss_history.append(epoch_mse)

    # --- VALIDATION STEP ---
    model.eval()  # Switch to eval mode
    with torch.no_grad():
        val_preds = model(X_val_device)
        val_loss = criterion(val_preds, y_val_device)
        val_mse = val_loss.item()
        val_rmse = np.sqrt(val_mse)
        val_loss_history.append(val_mse)

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch + 1:<8} {epoch_mse:<15.6f} {epoch_rmse:<15.6f} {val_mse:<15.6f} {val_rmse:<15.6f}")

print("-" * 95)

# Evaluate on test set (Final Check)
model.eval()
with torch.no_grad():
    preds_test = model(X_val_device).cpu().numpy().flatten()

true_test = y_test_tensor.numpy().flatten()
mse = mean_squared_error(true_test, preds_test)
rmse = np.sqrt(mse)

print(f"\nFinal Test Results:")
print(f"Test MSE:  {mse:.6f}")
print(f"Test RMSE: {rmse:.6f}")

# Save true and predicted values into a DataFrame
results_df = pl.DataFrame({
    "true_next_q1_log1p": true_test,
    "pred_next_q1_log1p": preds_test,
})
print(results_df.head())

# Plot training and validation loss
plt.figure(figsize=(6, 4))
plt.plot(loss_history, label='Train MSE')
plt.plot(val_loss_history, label='Val MSE')  # Added Validation plot
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot true vs predicted on test set
plt.figure(figsize=(6, 6))
plt.scatter(true_test, preds_test, alpha=0.5)
min_val = min(true_test.min(), preds_test.min())
max_val = max(true_test.max(), preds_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
plt.xlabel("True Next_Q1_log1p")
plt.ylabel("Predicted Next_Q1_log1p")
plt.title(f"True vs Predicted on Test Set (RMSE={rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()
