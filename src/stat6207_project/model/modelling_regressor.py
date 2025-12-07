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


def apply_imputation(df_target, source_medians, fallback_medians):
    """
    Applies series-specific medians.
    If a series is unknown (NaN), fills with global training median.
    """
    # Join the target with the learned medians
    # We use 'left' to keep the target rows intact
    df_merged = df_target.join(source_medians, on="series", rsuffix="_median")

    for col in impute_cols:
        median_col = f"{col}_median"

        # Coalesce: 1. Original Value -> 2. Series Median -> 3. Global Median
        df_target[col] = df_target[col].fillna(df_merged[median_col]).fillna(fallback_medians[col])

    return df_target

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

    series_path = data_folder / "target_series_new_with_features.csv"
    df_full = pd.read_csv(series_path, dtype={
        "isbn": "string",
        "print_length": "float32",
        "number_of_reviews": "float32",
        "length": "float32",
        "item_weight": "float32",
        "width": "float32",
        "height": "float32",
        "rating": "float32",
        "price": "float32",
        "avg_discount_rate": "float32"
    })

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

    impute_cols = ["print_length", "length", "width", "height", "rating", "item_weight", "price"]

    # Capture raw train data before any imputation or transformation
    transform_cols = [
        "q_since_first", "avg_discount_rate", "print_length", "length", "width", "height", "rating", "price"
    ]
    raw_train = df_train[transform_cols].copy()

    # 1. LEARN: Calculate medians only on Training data
    # We group by series to get specific stats, and also get global stats for fallbacks
    series_medians = df_train.groupby("series")[impute_cols].median()
    global_medians = df_train[impute_cols].median()

    # 2. APPLY: Transform all sets using the knowledge from Train
    print("Imputing missing values...")
    df_train = apply_imputation(df_train.copy(), series_medians, global_medians)
    df_val = apply_imputation(df_val.copy(), series_medians, global_medians)
    df_test = apply_imputation(df_test.copy(), series_medians, global_medians)

    # Verify purity
    print(f"Missing values in Train after impute: {df_train[impute_cols].isna().sum().sum()}")
    print(f"Missing values in Test  after impute: {df_test[impute_cols].isna().sum().sum()}")

    # ------------------------------------------------------------------
    # Resume existing pipeline
    # ------------------------------------------------------------------

    # Now continue with your transform_cols logic...
    show = False
    for col in transform_cols:
        # Log1p transform
        train_np = np.log1p(df_train[col].to_numpy())
        val_np = np.log1p(df_val[col].to_numpy())
        test_np = np.log1p(df_test[col].to_numpy())


        df_train[col].plot.box()
        plt.title(f"Boxplot of {col} after log1p transformation")
        if show:
            plt.show()

        train_mean = train_np.mean()
        train_std = train_np.std()
        # Standardize
        train_np = (train_np - train_mean) / train_std
        val_np = (val_np - train_mean) / train_std
        test_np = (test_np - train_mean) / train_std

        # Clip using +3/-3 stddev
        train_np = np.clip(train_np, -3, 3)
        val_np = np.clip(val_np, -3, 3)
        test_np = np.clip(test_np, -3, 3)

        df_train[col] = train_np
        df_val[col] = val_np
        df_test[col] = test_np

        df_val[col].plot.box()
        plt.title(f"Boxplot of {col} in Val set after log1p and standardization")
        if show:
            plt.show()
        df_test[col].plot.box()
        plt.title(f"Boxplot of {col} in Test set after log1p and standardization")
        if show:
            plt.show()

    # New: Plot side-by-side boxplots for each transformed column (train set only)
    # Raw (pre-impute/transform) vs. Final Preprocessed
    for col in transform_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)  # Different scales, as raw and transformed differ

        raw_train[col].plot.box(ax=axes[0])
        axes[0].set_title(f"Raw {col} (Train Set)")
        axes[0].set_ylabel(col)

        df_train[col].plot.box(ax=axes[1])
        axes[1].set_title(f"Preprocessed {col} (Train Set)")
        axes[1].set_ylabel("Transformed Value")

        plt.suptitle(f"Preprocessing Effect: {col}", fontsize=14)
        plt.tight_layout()
        plt.show()

    print(f"Train rows : {len(df_train):,}")
    print(f"Val   rows : {len(df_val):,}")
    print(f"Test  rows : {len(df_test):,}")

    # ------------------------------------------------------------------
    # Convert ONLY the feature columns to tensors
    # ------------------------------------------------------------------

    target_col = "quantity"
    metadata_cols = ["isbn", "title", "year_quarter", "series"]
    drop_cols = ["price"]
    feat_cols = [c for c in df_full.columns if c not in (metadata_cols + [target_col])]

    X_train = torch.from_numpy(df_train[feat_cols].to_numpy().astype(np.float32))
    X_val = torch.from_numpy(df_val[feat_cols].to_numpy().astype(np.float32))
    X_test = torch.from_numpy(df_test[feat_cols].to_numpy().astype(np.float32))

    # Step 1: Log-transform the target in all splits
    qty_train_log = np.log1p(df_train[target_col])
    qty_val_log = np.log1p(df_val[target_col])
    qty_test_log = np.log1p(df_test[target_col])

    # Step 2: Fit StandardScaler only on training (log-transformed) targets
    target_train_mean = qty_train_log.mean()
    target_train_std = qty_train_log.std()  # usually very close to 1 after log1p

    # Step 3: Standardize all splits with training statistics
    y_train = ((qty_train_log - target_train_mean) / target_train_std).astype(np.float32).to_numpy()
    y_val = ((qty_val_log - target_train_mean) / target_train_std).astype(np.float32).to_numpy()
    y_test = ((qty_test_log - target_train_mean) / target_train_std).astype(np.float32).to_numpy()

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

    target_books_new = pd.read_csv(data_folder / "target_books_new.csv", dtype={"isbn": "string"})
    model.load_state_dict(torch.load("results/regressor_best.pth"))
    model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(
            target_books_new[feat_cols].to_numpy().astype(np.float32)
        ).to(device)
        preds_all = model(X_all).cpu().numpy()
    target_books_new["pred_quantity"] = np.expm1(preds_all * target_train_std + target_train_mean)
    print(target_books_new["pred_quantity"].to_numpy().reshape(-1, 1))

    # Optional: Save the model
    # torch.save(model.state_dict(), results_folder / "regressor_weights.pth")
    # print("Training complete. Model saved.")
    print("end")