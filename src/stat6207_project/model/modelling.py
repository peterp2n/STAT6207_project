import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pathlib import Path
import copy  # Added to deepcopy the model state
from autoencoder import AutoEncoder
from autoencoder_trainer import AutoEncoderTrainer

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

def oversampling(mask, target_ratio, replace, random_state, frac):
    majority_df = train_df[~mask]
    minority_df = train_df[mask]

    target_n = int(len(majority_df) / target_ratio)

    print(f"Upsampling minority from {len(minority_df)} to {target_n}...")

    minority_upsampled = minority_df.sample(
        n=target_n,
        replace=replace,
        random_state=random_state
    )

    # Combine and shuffle — identical to your original code
    new_train_df = pd.concat([majority_df, minority_upsampled])
    new_train_df = new_train_df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    return new_train_df


def clip_outliers(train_series: pd.Series, test_series: pd.Series) -> pd.Series:
    train_np = train_series.to_numpy()
    col_mean = train_np.mean()
    col_std = train_np.std()
    col_lower = col_mean - 3 * col_std
    col_upper = col_mean + 3 * col_std

    test_np = test_series.to_numpy()
    return (
        pd.Series(np.clip(train_np, col_lower, col_upper), index=train_series.index),
        pd.Series(np.clip(test_np, col_lower, col_upper), index=test_series.index)
    )

torch.manual_seed(42)


if __name__ == "__main__":

    # -------------------------------
    # Configuration & Paths
    # -------------------------------
    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)

    train_path = data_folder / "train_all_cols_v3.csv"
    test_path = data_folder / "test_all_cols_v3.csv"

    # -------------------------------
    # Load Data
    # -------------------------------
    train_df = pd.read_csv(train_path, dtype={"isbn": "string"})
    test_df = pd.read_csv(test_path, dtype={"isbn": "string"})

    # Filter for valid ISBNs first
    train_df = train_df.loc[train_df["isbn"].str.startswith(r"978")]
    test_df = test_df[test_df["isbn"].str.startswith(r"978")]

    TARGET_COL = "Next_Q1"
    OPP_TARGET_COL = "Next_Q1_log1p"

    # Now y_train_full will contain the targets for the oversampled data
    y_train_full = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    cols_use = [
        'print_length', 'item_weight', 'length', 'width',
           'height', 'channel',
           'Quarters_since_first', 'Previous_quarter_qty', 'Current_quarter_qty',
           'Book_Flag',
           'Avg_discount_cleaned', 'book_format_board_book',
           'book_format_cards', 'book_format_hardcover',
           'book_format_library_binding', 'book_format_paperback',
           'reading_age_adolescence or above', 'reading_age_baby',
           'reading_age_preadolescence', 'reading_age_preschool',
           'reading_age_toddler', 'Quarter_num_1', 'Quarter_num_2',
           'Quarter_num_3', 'Quarter_num_4'
    ]

    cols_drop = [col for col in train_df.columns if col not in cols_use + [TARGET_COL, OPP_TARGET_COL]]

    X_train_full = train_df.drop(columns=cols_drop)
    X_test = test_df.drop(columns=cols_drop)

    numeric_cols = X_train_full.select_dtypes(include='number').columns.tolist()
    numeric_cols = [col for col in numeric_cols if all(("quarter" not in col.lower(), "reading_age" not in col.lower(),
                                                        "channel" not in col.lower(), "book_format" not in col.lower()))]

    for col in numeric_cols:
        # Check if all values are non-negative
        if (X_train_full[col] >= 0).all():
            # Log transform first
            X_train_full[col] = np.log1p(X_train_full[col])
            X_test[col] = np.log1p(X_test[col])

            srs = clip_outliers(X_train_full[col], X_test[col])

    # Ensure identical column order
    X_test = X_test[X_train_full.columns]

    # -------------------------------
    # Train / Validation Split (80/20)
    # -------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, train_size=0.85, random_state=42
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

    # Move validation & test tensors to device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_val_device = X_val_tensor.to(device)
    y_val_device = y_val_tensor.to(device)
    X_test_device = X_test_tensor.to(device)

    batch_size = 512
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------
    # DataLoader
    # -------------------------------
    epochs = 250
    drop = 0.2
    learning_rate = 1e-5
    wt_decay = 0

    input_dim = X_train_tensor.shape[1]
    model = Regressor(input_dim=input_dim, dropout=drop).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    # -------------------------------
    # Training Loop with Validation
    # -------------------------------

    loss_history = []  # Train MSE per epoch
    val_loss_history = []  # Validation MSE per epoch
    clip_value = 1.0  # Threshold for gradient clipping

    # Best Model Tracking
    best_val_mse = float('inf')
    best_model_state = None
    best_epoch = -1

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
            loss = criterion(preds, yb)
            loss.backward()

            # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        train_mse = epoch_loss / len(train_loader.dataset)
        train_rmse = np.sqrt(train_mse)
        loss_history.append(train_rmse)

        # ---------- Validation ----------
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_device)
            val_loss = criterion(val_preds, y_val_device)
            val_mse = val_loss.item()
            val_rmse = np.sqrt(val_mse)
            val_loss_history.append(val_rmse)

        # --- Save Best Model Logic (MOVED HERE) ---
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            # Use deepcopy to ensure we store the actual weights, not just a reference
            best_model_state = copy.deepcopy(model.state_dict())
            # Optional: Print only when a new best is found
            # print(f"  -> New best model saved (Val MSE: {val_mse:.6f})")

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch + 1:<8} {train_mse:<15.6f} {train_rmse:<15.6f} {val_mse:<15.6f} {val_rmse:<15.6f}")

    print("-" * 95)
    print(f"Restoring best model from Epoch {best_epoch} with Val MSE: {best_val_mse:.6f}")

    # --- Load Best Weights BEFORE Testing ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------------------
    # Final Test Evaluation
    # -------------------------------
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test_device).cpu().numpy().flatten()

    true_test = y_test_tensor.numpy().flatten()

    test_mse = mean_squared_error(true_test, preds_test)
    test_rmse = np.sqrt(test_mse)

    print(f"\nFinal Test Results (Best Model):")
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
    plt.plot(loss_history, label="Train RMSE", linewidth=2)
    plt.plot(val_loss_history, label="Validation RMSE", linewidth=2)
    # Add a marker for the best epoch
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.ylim(bottom=min(min(loss_history), min(val_loss_history)) * 0.9 , top=max(max(loss_history), max(val_loss_history)) * 1.1)
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

    #==========================
    target_path = data_folder / "target_books_cleaned_v2.csv"
    target_df = pd.read_csv(target_path, dtype={"isbn": "string"})

    target_mean = y_train_full.mean()
    target_std = y_train_full.std()
    # Use exactly the same feature columns and order as training
    X_target = target_df[X_train_full.columns]

    # To tensor and device
    X_target_tensor = torch.tensor(X_target.to_numpy(), dtype=torch.float32).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        preds_target = model(X_target_tensor).cpu().numpy().flatten()
    print("Predicted quantity: ", preds_target)

    # preds_target = np.expm1(preds_target)
    target_col_name = "pred_next_q1"
    preds_target_df = pd.concat([target_df, pd.DataFrame({target_col_name: preds_target})], axis=1)
    preds_target_df = preds_target_df[["isbn", target_col_name]]
    # preds_target_df.to_csv(data_folder / "target_pred_q1_v2.csv", index=False)
    # preds_target_df.to_excel(data_folder / "target_pred_q1_v2.xlsx", index=False)

    # preds_target = preds_target * target_std + target_mean

    # 2) log1p(x) -> x
    next_q1_pred = np.expm1(preds_target)
    print("Reversed prediction: ", next_q1_pred)

    reverse_mean = np.expm1(target_mean)


