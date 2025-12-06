from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from regressor import Regressor

device = torch.device("mps")

def log_and_plot_regression_history(
    epoch: int,
    train_mse: float,
    val_mse: float,
    train_history: list,
    val_history: list,
    best_val_rmse: float = None,
    is_best: bool = False,
    print_every: int = 10
):
    """
    Logs metrics every epoch (for plotting), but only prints every `print_every` epochs.
    """
    train_rmse = np.sqrt(train_mse)
    val_rmse   = np.sqrt(val_mse)

    # Always store for plotting
    train_history.append(train_rmse)
    val_history.append(val_rmse)

    # Only print every N epochs or when it's the best model
    should_print = (epoch + 1) % print_every == 0 or is_best or (epoch + 1) == 100  # assuming max 100 epochs

    if should_print:
        status = " BEST" if is_best else ""
        print(
            f"Epoch {epoch+1:3d} | "
            f"Train MSE: {train_mse:8.5f}  RMSE: {train_rmse:7.4f} | "
            f"Val   MSE: {val_mse:8.5f}  RMSE: {val_rmse:7.4f}{status}"
        )

    # Return updated best RMSE
    if best_val_rmse is None:
        return val_rmse
    return min(best_val_rmse, val_rmse)


def plot_rmse_curve(train_history: list, val_history: list, best_epoch: int = None):
    """
    Call once after training finishes to display the gorgeous RMSE curve.
    """
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_history) + 1)

    plt.plot(epochs_range, train_history, label="Training RMSE", linewidth=2.5, color="#1f77b4")
    plt.plot(epochs_range, val_history,   label="Validation RMSE", linewidth=2.5, color="#ff7f0e")

    if best_epoch is not None:
        plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.7,
                    label=f"Best model (epoch {best_epoch})")

    plt.title("Book Sales Regressor — RMSE over Epochs", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (log1p + standardized quantity)")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)

    results_folder = Path("results")
    encoder_path = results_folder / "encoder_weights.pth"

    series_path = data_folder / "target_series_new.csv"
    df_full = pd.read_csv(series_path, dtype={"isbn": "string"})

    df_train, df_temp = train_test_split(
        df_full,
        test_size=0.3,
        random_state=42,
        shuffle=True,
        stratify=None,  # you can add stratification on e.g. year/quarter if desired
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,  # 15% of total each
        random_state=42,
        shuffle=True,
    )

    transform_cols = [
        "q_since_first", "discount_rate"
    ]

    show = False
    for col in transform_cols:
        df_train[col] = np.log1p(df_train[col])
        df_train[col].plot.box()
        plt.title(f"Boxplot of {col} after log1p transformation")
        if show:
            plt.show()

        train_mean = df_train[col].mean()
        train_std = df_train[col].std()

        df_val[col] = (np.log1p(df_val[col]) - train_mean) / train_std
        df_test[col] = (np.log1p(df_test[col]) - train_mean) / train_std

        df_val[col].plot.box()
        plt.title(f"Boxplot of {col} in Val set after log1p and standardization")
        if show:
            plt.show()
        df_test[col].plot.box()
        plt.title(f"Boxplot of {col} in Test set after log1p and standardization")
        if show:
            plt.show()

    print(f"Train rows : {len(df_train):,}")
    print(f"Val   rows : {len(df_val):,}")
    print(f"Test  rows : {len(df_test):,}")

    # ------------------------------------------------------------------
    # Convert ONLY the feature columns to tensors
    # ------------------------------------------------------------------

    target_col = "quantity"
    metadata_cols = ["isbn", "title"]  # keep these for later merging
    feat_cols = [c for c in df_full.columns if c not in (metadata_cols + [target_col])]

    X_train = torch.from_numpy(df_train[feat_cols].values.astype(np.float32))
    X_val = torch.from_numpy(df_val[feat_cols].values.astype(np.float32))
    X_test = torch.from_numpy(df_test[feat_cols].values.astype(np.float32))

    # Step 1: Log-transform the target in all splits
    qty_train_log = np.log1p(df_train[target_col])
    qty_val_log = np.log1p(df_val[target_col])
    qty_test_log = np.log1p(df_test[target_col])

    # Step 2: Fit StandardScaler only on training (log-transformed) targets
    train_mean = qty_train_log.mean()
    train_std = qty_train_log.std()  # usually very close to 1 after log1p

    # Step 3: Standardize all splits with training statistics
    y_train = ((qty_train_log - train_mean) / train_std).astype(np.float32).to_numpy()
    y_val = ((qty_val_log - train_mean) / train_std).astype(np.float32).to_numpy()
    y_test = ((qty_test_log - train_mean) / train_std).astype(np.float32).to_numpy()

    # Convert to tensors as before
    y_train = torch.from_numpy(y_train).to(device)
    y_val = torch.from_numpy(y_val).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    # Move to device
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    y_test = y_test.to(device)

    print(f"Train tensor : {X_train.shape}")
    print(f"Val   tensor : {X_val.shape}")
    print(f"Test  tensor : {X_test.shape}")

    input_dim = X_train.shape[1]

    # # ======== Load the trained encoder ========
    #
    # encoder = load_encoder_weights(
    #     input_dim=input_dim,
    #     encoding_dim=6,
    #     weights_path=encoder_path,
    #     device=device
    # )
    # print("Trained encoder loaded and ready for embedding extraction")
    batch_size = 128
    epochs = 50
    learning_rate = 0.001
    drop = 0.5
    decay = 1e-3

    # Training the regressor
    model = Regressor(input_dim=input_dim, dropout=drop).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_rmses = []
    val_rmses = []
    best_val_rmse = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        train_rmse = np.sqrt(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_ds)
        val_rmse = np.sqrt(val_loss)

        is_best = False
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            is_best = True
            torch.save(model.state_dict(), "results/regressor_best.pth")

        # This now handles BOTH logging AND smart printing
        log_and_plot_regression_history(
            epoch=epoch,
            train_mse=train_loss,
            val_mse=val_loss,
            train_history=train_rmses,
            val_history=val_rmses,
            best_val_rmse=best_val_rmse,
            is_best=is_best,
            print_every=10
        )


    # Optional: Evaluate on test set after training
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = loss_fn(test_preds, y_test).item()
    test_rmse = np.sqrt(test_loss)
    print(f"Test MSE {test_loss:.4f}, RMSE {test_rmse:.4f}")

    # After loop
    plot_rmse_curve(train_rmses, val_rmses, best_epoch=best_epoch)

    target_books_new = pd.read_csv(data_folder / "target_series_new.csv", dtype={"isbn": "string"})
    model.load_state_dict(torch.load("results/regressor_best.pth"))
    model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(
            target_books_new[feat_cols].values.astype(np.float32)
        ).to(device)
        preds_all = model(X_all).cpu().numpy()
    target_books_new["pred_quantity"] = np.expm1(preds_all * preds_all.std() + preds_all.mean())
    print(target_books_new["pred_quantity"].to_numpy().reshape(-1, 1))

    # Optional: Save the model
    # torch.save(model.state_dict(), results_folder / "regressor_weights.pth")
    # print("Training complete. Model saved.")
    print("end")