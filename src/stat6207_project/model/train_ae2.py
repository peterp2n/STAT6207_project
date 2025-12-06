# train_ae2.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from autoencoder_trainer import AutoEncoderTrainer
import matplotlib.pyplot as plt

# ================================
# 1. Set up paths and device
# ================================
data_folder = Path("data")
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

# Auto-detect device: MPS if available (Apple Silicon), else CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# ================================
# 2. Load and prepare data
# ================================
print("Loading data...")

# Full dataset (features + target + metadata)
data_path = data_folder / "target_series_new.csv"
df_full = pd.read_csv(data_path, dtype={"isbn": str})          # keep isbn as string

print(f"Loaded {len(df_full):,} rows with {df_full.shape[1]} columns")

# ------------------------------------------------------------------
# Define target and feature columns
# ------------------------------------------------------------------
target_col = "quantity"
metadata_cols = ["isbn", "title"]          # keep these for later merging
feat_cols = [c for c in df_full.columns if c not in (metadata_cols + [target_col])]



print(f"Using {len(feat_cols)} feature columns for the autoencoder")
# print(feat_cols)   # uncomment if you want to double-check

# ------------------------------------------------------------------
# Handle missing values (same strategy as before)
# ------------------------------------------------------------------
df_full[feat_cols] = (
    df_full[feat_cols]
    .fillna(method="ffill")
    .fillna(method="bfill")
    .fillna(0)
)

# ------------------------------------------------------------------
# Train / Val / Test split on the *DataFrame* level first
# (this guarantees you can always recover the original rows/isbn/title)
# ------------------------------------------------------------------
# 70% train, 15% val, 15% test  →  total test_size=0.3, then half of that for val
df_train, df_temp = train_test_split(
    df_full,
    test_size=0.3,
    random_state=42,
    shuffle=True,
    stratify=None,          # you can add stratification on e.g. year/quarter if desired
)

df_val, df_test = train_test_split(
    df_temp,
    test_size=0.5,          # 15% of total each
    random_state=42,
    shuffle=True,
)

transform_cols = [
    "quantity", "q_since_first", "discount_rate"
]
for col in transform_cols:
    df_train[col] = np.log1p(df_train[col])
    df_train[col].plot.box()
    plt.title(f"Boxplot of {col} after log1p transformation")
    plt.show()

    train_mean = df_train[col].mean()
    train_std = df_train[col].std()

    df_val[col] = (np.log1p(df_val[col]) - train_mean) / train_std
    df_test[col] = (np.log1p(df_test[col]) - train_mean) / train_std

    df_val[col].plot.box()
    plt.title(f"Boxplot of {col} in Val set after log1p and standardization")
    plt.show()
    df_test[col].plot.box()
    plt.title(f"Boxplot of {col} in Test set after log1p and standardization")
    plt.show()



print(f"Train rows : {len(df_train):,}")
print(f"Val   rows : {len(df_val):,}")
print(f"Test  rows : {len(df_test):,}")

# ------------------------------------------------------------------
# Convert ONLY the feature columns to tensors
# ------------------------------------------------------------------
X_train = torch.from_numpy(df_train[feat_cols].values.astype(np.float32))
X_val   = torch.from_numpy(df_val[feat_cols].values.astype(np.float32))
X_test  = torch.from_numpy(df_test[feat_cols].values.astype(np.float32))

# Move to device
X_train = X_train.to(device)
X_val   = X_val.to(device)
X_test  = X_test.to(device)

print(f"Train tensor : {X_train.shape}")
print(f"Val   tensor : {X_val.shape}")
print(f"Test  tensor : {X_test.shape}")

input_dim = X_train.shape[1]

# ================================
# 3. Initialize and train autoencoder
# ================================
print("\nInitializing AutoEncoderTrainer...")

trainer = AutoEncoderTrainer(
    input_dim=input_dim,
    encoding_dim=6,
    lr=1e-4
)

print(f"Starting training on {device}...")
trainer.train(
    train_data=X_train,
    val_data=X_val,
    epochs=100,
    batch_size=64,
    print_every=10
)

# ================================
# 4. Final evaluation
# ================================
test_mse, test_rmse = trainer.evaluate(X_test)

print(f"\nFinal Test Results:")
print(f"   Test MSE  : {test_mse:.6f}")
print(f"   Test RMSE : {test_rmse:.6f}")

# ================================
# 5. Plot training curves
# ================================
trainer.plot_losses(
    title="Book Features Autoencoder - Training & Validation Loss",
    save_path=results_folder / "autoencoder_loss_curve.png",
    y_scale='auto',
    y_lim="auto"
)

# ================================
# 6. Save encoder weights
# ================================
encoder_path = results_folder / "encoder_weights.pth"
# trainer.save_weights(part="encoder", path=encoder_path)


# Optional: reconstruction similarity check
print("Computing reconstruction similarity on training set...")
cos_sim = trainer.get_reconstruction_similarity(X_train)
print(f"Mean cosine similarity (train): {cos_sim.mean():.4f} ± {cos_sim.std():.4f}")

print("\nTraining complete! Results saved in:", results_folder)