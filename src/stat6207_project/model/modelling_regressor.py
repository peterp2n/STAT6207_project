from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from regressor import Regressor
import random


def set_seed(seed_value=42):
    """Set seeds for reproducibility across all libraries."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

    # If using CUDA/GPU, ensure deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # If using MPS/Apple Silicon
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)

def apply_imputation(df_target, source_medians, fallback_medians, impute_cols):
    """
    Applies series-specific medians for selected columns.
    """
    # Join the target with the learned medians
    df_merged = df_target.join(source_medians, on="series", rsuffix="_median")

    for col in impute_cols:
        median_col = f"{col}_median"
        # Coalesce: 1. Original Value -> 2. Series Median -> 3. Global Median
        df_target[col] = df_target[col].fillna(df_merged[median_col]).fillna(fallback_medians[col])

    return df_target


def log_and_plot_regression_history(epoch, train_mse, val_mse, train_hist, val_hist, best_rmse=None, is_best=False):
    train_rmse, val_rmse = np.sqrt(train_mse), np.sqrt(val_mse)
    train_hist.append(train_rmse);
    val_hist.append(val_rmse)

    if (epoch + 1) % 10 == 0 or is_best:
        status = " BEST" if is_best else ""
        print(f"Epoch {epoch + 1:3d} | Train MSE: {train_mse:.5f} RMSE: {train_rmse:.4f} | "
              f"Val MSE: {val_mse:.5f} RMSE: {val_rmse:.4f}{status}")

    if best_rmse is None: return val_rmse
    return min(best_rmse, val_rmse)


def plot_rmse_curve(train_history, val_history, best_epoch=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label="Training RMSE", linewidth=2.5, color="#1f77b4")
    plt.plot(val_history, label="Validation RMSE", linewidth=2.5, color="#ff7f0e")
    if best_epoch: plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best (ep {best_epoch})")
    plt.title("Book Sales Regressor — RMSE", fontsize=18, pad=20)
    plt.legend();
    plt.grid(True, alpha=0.3);
    plt.tight_layout();
    plt.show()


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    set_seed(42)
    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)

    # Load Data
    series_path = data_folder / "target_series_new_with_features2.csv"
    df_full = pd.read_csv(series_path, dtype={
        "isbn": "string", "print_length": "float32", "number_of_reviews": "float32",
        "length": "float32", "item_weight": "float32", "width": "float32",
        "height": "float32", "rating": "float32", "price": "float32",
        "avg_discount_rate": "float32"
    })

    df_train, df_temp = train_test_split(df_full, test_size=0.2, random_state=42, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

    # ------------------------------------------------------------------
    # 1. Configuration: Imputation Split
    # ------------------------------------------------------------------
    # We are using Series Median for everything.

    # List of columns to impute missing values using median (series then global fallback)
    impute_cols = [
        "length",
        "width",
        "rating",
        "item_weight",
        "price",
        "print_length",
        "height"
    ]

    # List of all continuous columns that will undergo the full transformation pipeline (log1p, robust scale, clip)
    transform_cols = [
        "q_since_first",
        "avg_discount_rate",
        "print_length",
        "length",
        "width",
        "height",
        "rating",
        "price"
    ]

    # List of columns that should receive the log1p transformation (subset of transform_cols)
    log_cols = [
        "q_since_first",
        "length",
        "width",
        "height",
        "price"
    ]

    # List of columns to apply clipping after scaling (subset of transform_cols)
    # 'quantity' target variable should ideally be handled separately if it follows Pareto distribution
    clip_cols = [
        "avg_discount_rate",
        "rating",
        "item_weight",
        "q_since_first"
    ]

    raw_train = df_train[transform_cols + ["quantity"]].copy()  # Capture for Before Heatmap/Boxplots

    # 2. Learn Medians (Train only)
    series_medians = df_train.groupby("series")[impute_cols].median()
    global_medians = df_train[impute_cols].median()

    # 3. Apply Imputation
    print("Imputing missing values...")
    for df in [df_train, df_val, df_test]:
        apply_imputation(df, series_medians, global_medians, impute_cols)

    # ------------------------------------------------------------------
    # 4. Transformation Logic
    # ------------------------------------------------------------------

    # NEW: Dictionary to store the fitted RobustScaler objects
    scalers = {}
    transformed_train = {}  # Used to store transformed training data for potential analysis/visualization

    print("\nTransforming features...")
    for col in transform_cols:
        # Ensure columns are treated as float for numpy/sklearn compatibility
        train_np = df_train[col].to_numpy().astype(np.float32)
        val_np = df_val[col].to_numpy().astype(np.float32)
        test_np = df_test[col].to_numpy().astype(np.float32)

        # Check config lists
        is_log = col in log_cols
        is_clipped = col in clip_cols

        # Define clipping thresholds (using wider range as discussed, or original 5, 5 for certain features)
        CLIP_MIN = -7
        CLIP_MAX = 7

        # 1. Log Transform (if applicable)
        if is_log:
            train_np = np.log1p(train_np)
            val_np = np.log1p(val_np)
            test_np = np.log1p(test_np)

        # 2. Robust Scaling (Fit only on train data)
        # We MUST save this scaler to apply the exact same median/IQR to validation and test sets
        scaler = RobustScaler()
        # Scikit-learn requires 2D array input, reshape to (-1, 1)
        train_np = scaler.fit_transform(train_np.reshape(-1, 1)).flatten()
        val_np = scaler.transform(val_np.reshape(-1, 1)).flatten()
        test_np = scaler.transform(test_np.reshape(-1, 1)).flatten()

        # Store the scaler for later use
        scalers[col] = scaler

        # 3. Clip Predictors (if applicable)
        # Clip X to prevent gradient instability for specific features
        if is_clipped:
            train_np = np.clip(train_np, CLIP_MIN, CLIP_MAX)
            val_np = np.clip(val_np, CLIP_MIN, CLIP_MAX)
            test_np = np.clip(test_np, CLIP_MIN, CLIP_MAX)

        # Assign back to DataFrames
        df_train[col] = train_np
        df_val[col] = val_np
        df_test[col] = test_np
        transformed_train[col] = train_np

    # ------------------------------------------------------------------
    # Target Transformation for Visualization & Consistency (RobustScaler)
    # ------------------------------------------------------------------
    print("Fitting target RobustScaler on training data for consistent visualization...")

    # Fit the same y_scaler that will be used later in training
    y_scaler = RobustScaler()
    qty_train_log = np.log1p(df_train["quantity"].values.reshape(-1, 1))
    y_train_scaled_for_viz = y_scaler.fit_transform(qty_train_log)  # Shape: (n, 1)

    # Extract center (median) and scale (IQR) for future inverse transform
    y_center = y_scaler.center_[0]
    y_scale = y_scaler.scale_[0]

    # Flatten and clip for visualization consistency (same as model input)
    qty_trans = y_train_scaled_for_viz.flatten()

    # Store for correlation heatmap and scatterplots
    transformed_train["quantity"] = qty_trans

    print(f"Target transformed: log(quantity) → RobustScaler (median={y_center:.4f}, IQR={y_scale:.4f})")

    # ------------------------------------------------------------------
    # HEATMAP: Now Using True Model Target (log1p + RobustScaler + Clipped)
    # ------------------------------------------------------------------
    corr_before = raw_train.corr()
    corr_after = pd.DataFrame(transformed_train).corr()

    fig, axes = plt.subplots(1, 2, figsize=(32, 14))
    heatmap_args = {"annot": True, "fmt": ".2f", "cmap": "coolwarm", "center": 0, "square": True}

    sns.heatmap(corr_before, ax=axes[0], **heatmap_args)
    axes[0].set_title("Correlation: Raw Train Data (Before)", fontsize=18)

    sns.heatmap(corr_after, ax=axes[1], **heatmap_args)
    axes[1].set_title("Correlation: After log1p + Robust Scaling (Outlier-Resistant)", fontsize=18)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # SCATTERPLOTS: Feature vs Quantity (The Reality Check)
    # ------------------------------------------------------------------
    # We compare the chaotic raw world (Blue) vs the organized model world (Orange)
    scatter_features = ["print_length", "height"]

    # Ensure we have the transformed quantity for plotting
    # (It was calculated in the Heatmap section as 'qty_trans')

    print("\nGenerating Scatterplots: Feature vs Quantity...")

    for col in scatter_features:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. The Raw World (Blue)
        # Using pure raw data. This shows us the outliers and the skew.
        axes[0].scatter(raw_train[col], raw_train["quantity"],
                        alpha=0.6, c='#1f77b4', edgecolors='none', s=15)
        axes[0].set_title(f"Raw Reality: {col} vs Quantity", fontsize=14, fontweight='medium')
        axes[0].set_xlabel(f"Raw {col}")
        axes[0].set_ylabel("Raw Quantity")
        axes[0].grid(True, linestyle='--', alpha=0.3)

        # 2. The Transformed World (Orange)
        # Using the processed data (df_train) and the transformed quantity (qty_trans)
        # This is exactly what the neural network sees.
        axes[1].scatter(df_train[col], transformed_train["quantity"],
                        alpha=0.6, c='#ff7f0e', edgecolors='none', s=15)
        axes[1].set_title(f"Model View: Robust-Scaled {col} vs Target", fontsize=14, fontweight='medium')
        axes[1].set_xlabel(f"Robust-Scaled {col} (IQR Units)")
        axes[1].set_ylabel("Robust-Scaled log(Quantity) (IQR Units)")
        axes[1].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Model Prep
    # ------------------------------------------------------------------
    # Stacked Bars
    df_train_viz = df_train.copy();
    df_train_viz["quantity_raw"] = raw_train["quantity"]
    for cat in ["format", "channel"]:
        (df_train_viz.groupby(["q_num", cat])["quantity_raw"].sum().unstack().fillna(0)
         .plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis", edgecolor="black"))
        plt.title(f"Sales by {cat}");
        plt.tight_layout();
        plt.show()

    # Tensors
    dummy_cols = ["format", "channel"]
    df_train = pd.get_dummies(df_train, columns=dummy_cols, drop_first=True)
    df_val = pd.get_dummies(df_val, columns=dummy_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=dummy_cols, drop_first=True)

    target_col = "quantity"
    meta_data_cols = [
        "isbn",
        "title",
        "year_quarter",
        "series"
    ]
    potential_feat_cols = [
        "price",
        "height",
        "length",
        "width",
        "item_weight",
        "print_length",
        "number_of_reviews",
        "rating",
        "q_since_first",
        "q_num",
        target_col
    ]

    drop_cols = [
        "price",
        "height",
        "length",
        "width",
        "item_weight",
        "print_length",
        "number_of_reviews",
        "rating",
        "q_since_first",
        "q_num",
        target_col
    ]
    feat_cols = [c for c in df_train.columns if c not in meta_data_cols + drop_cols + [target_col]]

    print(f"Features being used: {feat_cols}")

    X_train = torch.from_numpy(df_train[feat_cols].to_numpy().astype(np.float32)).to(device)
    X_val = torch.from_numpy(df_val[feat_cols].to_numpy().astype(np.float32)).to(device)
    X_test = torch.from_numpy(df_test[feat_cols].to_numpy().astype(np.float32)).to(device)

    # ------------------------------------------------------------------
    # Target Transform (UPDATED: Robust Scaler)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Target Transform — Using Pre-Fitted y_scaler (Already Done for Visualization)
    # ------------------------------------------------------------------
    qty_val_log = np.log1p(df_val["quantity"].values.reshape(-1, 1))
    qty_test_log = np.log1p(df_test["quantity"].values.reshape(-1, 1))

    # Transform val/test using the already-fitted scaler
    y_val_scaled = y_scaler.transform(qty_val_log).flatten()
    y_test_scaled = y_scaler.transform(qty_test_log).flatten()

    # Clip all targets consistently with training
    y_train = torch.from_numpy(y_train_scaled_for_viz.flatten().astype(np.float32)).to(device)
    y_val = torch.from_numpy(y_val_scaled.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test_scaled.astype(np.float32)).to(device)

    # ------------------------------------------------------------------
    # Print tensor dimensions before training
    # ------------------------------------------------------------------
    print("\n=== Dataset Tensor Shapes ===")
    print(f"X_train: {X_train.shape}  →  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}   →  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  →  y_test:  {y_test.shape}")
    print(f"Feature dimension used in model → {X_train.shape[1]}")
    print("=" * 34)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    drop = 0.5
    learning_rate = 0.001
    decay = 1e-3
    b_size = 128
    epochs = 100

    model = Regressor(input_dim=X_train.shape[1], dropout=drop).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    loss_fn = nn.MSELoss()

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=b_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=b_size, shuffle=False)

    train_rmses, val_rmses, best_rmse, best_ep = [], [], float('inf'), 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad();
            loss = loss_fn(model(xb), yb);
            loss.backward();
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in val_dl) / len(X_val)

        is_best = val_loss < best_rmse
        if is_best: best_rmse = val_loss; best_ep = epoch + 1; torch.save(model.state_dict(),
                                                                          "results/regressor_best.pth")
        log_and_plot_regression_history(epoch, train_loss, val_loss, train_rmses, val_rmses, np.sqrt(best_rmse),
                                        is_best)

    plot_rmse_curve(train_rmses, val_rmses, best_ep)

    # ------------------------------------------------------------------
    # Load Best Model & Evaluate on Test Set
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load("results/regressor_best.pth"))
    model.eval()

    test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=b_size, shuffle=False)
    test_loss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in test_dl) / len(X_test)
    test_rmse = np.sqrt(test_loss)

    print(f"\nTest MSE: {test_loss:.5f} RMSE: {test_rmse:.4f}")

    # ------------------------------------------------------------------
    # Final Prediction
    # ------------------------------------------------------------------
    target_books = pd.read_csv(data_folder / "target_books_new.csv", dtype={"isbn": "string"})
    final_ids = target_books[["isbn", "title"]].copy()

    # 1. Impute (Median where valid)
    target_books = apply_imputation(target_books, series_medians, global_medians, impute_cols)

    # 2. Transform (CORRECTED)
    for col in transform_cols:
        # Get raw values
        vals = target_books[col].to_numpy().astype(np.float32)

        # A. Log Transform (Check global list directly)
        if col in log_cols:
            vals = np.log1p(vals)

        # B. Robust Scale (Reuse the fitted scaler from training)
        if col in scalers:
            vals = scalers[col].transform(vals.reshape(-1, 1)).flatten()

        # C. Clip (Same threshold as training)
        target_books[col] = np.clip(vals, CLIP_MIN, CLIP_MAX)

    # 3. Predict
    target_books = pd.get_dummies(target_books, columns=dummy_cols, drop_first=True)
    X_final = torch.from_numpy(target_books.reindex(columns=feat_cols, fill_value=0.0).values.astype(np.float32)).to(
        device)

    model.eval()
    with torch.no_grad():
        preds = model(X_final).cpu().numpy().flatten()

    # Inverse Robust Scale: (preds * IQR) + Median
    pred_log = preds * y_scale + y_center

    # Inverse Log: expm1
    final_ids["pred_quantity"] = np.expm1(pred_log).round(0).astype(int)

    print("\nFinal Predictions:\n", final_ids.head(10))
    final_ids.to_csv("results/final_predictions.csv", index=False)
