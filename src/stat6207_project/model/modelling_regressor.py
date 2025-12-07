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
import copy

# ------------------------------------------------------------------
# Hyperparameters (Global Configuration)
# ------------------------------------------------------------------
DROPOUT = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
BATCH_SIZE = 128
EPOCHS = 100
CLIP_MIN = -7
CLIP_MAX = 7
SEED = 42

# Feature Configuration
IMPUTE_COLS = ["length", "width", "rating", "item_weight", "price", "print_length", "height"]
TRANSFORM_COLS = ["q_since_first", "avg_discount_rate", "print_length", "length",
                  "width", "height", "rating", "price", "item_weight"]
LOG_COLS = ["q_since_first", "length", "width", "height", "price"]
CLIP_COLS = ["avg_discount_rate", "rating", "item_weight", "q_since_first"]
DUMMY_COLS = ["format", "channel", "q_num"]
META_COLS = ["isbn", "title", "year_quarter", "series"]
TARGET_COL = "quantity"
POTENTIAL_FEAT_COLS = [
    "price", "height", "length", "width", "item_weight",
    "print_length", "rating", "q_since_first"
]


def set_seed(seed_value=SEED):
    """Set seeds for reproducibility across all libraries."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)


def apply_imputation(df_target, source_medians, fallback_medians, impute_cols):
    """
    Applies series-specific medians for selected columns.
    """
    df_merged = df_target.join(source_medians, on="series", rsuffix="_median")

    for col in impute_cols:
        median_col = f"{col}_median"
        df_target[col] = df_target[col].fillna(df_merged[median_col]).fillna(fallback_medians[col])

    return df_target


def greedy_forward_selection(potential_feat_cols, df_train, df_val, y_train, y_val,
                             meta_data_cols, target_col, device):
    """Greedy forward feature selection - O(n²) instead of O(2^n)."""
    selected = []
    remaining = list(potential_feat_cols)
    best_rmse_history = []

    # Get all available columns (including dummies)
    all_cols = [c for c in df_train.columns if c not in meta_data_cols + [target_col]]

    while remaining:
        best_rmse = float('inf')
        best_feature = None

        for feat in remaining:
            # Try adding this feature
            candidate = selected + [feat]

            # Build feature list (exclude what's NOT in candidate from potential_feat_cols)
            drop_cols = [c for c in potential_feat_cols if c not in candidate]
            feat_cols = [c for c in all_cols if c not in drop_cols]

            if not feat_cols:
                continue

            X_train = torch.from_numpy(df_train[feat_cols].to_numpy().astype(np.float32)).to(device)
            X_val = torch.from_numpy(df_val[feat_cols].to_numpy().astype(np.float32)).to(device)

            set_seed(SEED)
            rmse, _, _, _, _ = train_model(X_train, y_train, X_val, y_val, device, verbose=False)

            if rmse < best_rmse:
                best_rmse = rmse
                best_feature = feat

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_rmse_history.append((list(selected), best_rmse))
        print(f"Added '{best_feature}' → RMSE: {best_rmse:.4f} | Features: {selected}")

    # Find best subset from history
    best_idx = min(range(len(best_rmse_history)), key=lambda i: best_rmse_history[i][1])
    return best_rmse_history[best_idx][0], best_rmse_history


def greedy_backward_elimination(potential_feat_cols, df_train, df_val, y_train, y_val,
                                meta_data_cols, target_col, device):
    """Start with all features, remove one at a time."""
    selected = list(potential_feat_cols)
    all_cols = [c for c in df_train.columns if c not in meta_data_cols + [target_col]]
    best_rmse_history = []

    while len(selected) > 1:
        best_rmse = float('inf')
        worst_feature = None

        for feat in selected:
            # Try removing this feature
            candidate = [f for f in selected if f != feat]
            drop_cols = [c for c in potential_feat_cols if c not in candidate]
            feat_cols = [c for c in all_cols if c not in drop_cols]

            X_train = torch.from_numpy(df_train[feat_cols].to_numpy().astype(np.float32)).to(device)
            X_val = torch.from_numpy(df_val[feat_cols].to_numpy().astype(np.float32)).to(device)

            set_seed(SEED)
            rmse, _, _, _, _ = train_model(X_train, y_train, X_val, y_val, device, verbose=False)

            if rmse < best_rmse:
                best_rmse = rmse
                worst_feature = feat

        selected.remove(worst_feature)
        best_rmse_history.append((list(selected), best_rmse))
        print(f"Removed '{worst_feature}' → RMSE: {np.sqrt(best_rmse):.4f} | Remaining: {selected}")

    best_idx = min(range(len(best_rmse_history)), key=lambda i: best_rmse_history[i][1])
    return best_rmse_history[best_idx][0], best_rmse_history


def plot_actual_vs_predicted(model, X_test, y_test, y_scale, y_center, device):
    """
    Plot actual vs predicted quantity for test set with best fit line.
    Shows both transformed (scaled) and original scale in separate plots.
    """
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy().flatten()

    actual_scaled = y_test.cpu().numpy().flatten()

    pred_log = preds_scaled * y_scale + y_center
    pred_qty = np.expm1(pred_log)

    actual_log = actual_scaled * y_scale + y_center
    actual_qty = np.expm1(actual_log)

    # --- Plot 1: Transformed Scale ---
    coeffs_scaled = np.polyfit(preds_scaled, actual_scaled, 1)
    fit_scaled = np.poly1d(coeffs_scaled)
    x_range_scaled = np.linspace(preds_scaled.min(), preds_scaled.max(), 100)

    plt.figure(figsize=(10, 8))
    plt.scatter(preds_scaled, actual_scaled, alpha=0.5, c='#1f77b4', edgecolors='none', s=20, label='Test samples')
    plt.plot(x_range_scaled, fit_scaled(x_range_scaled), 'r--', linewidth=2,
             label=f'Best fit (y={coeffs_scaled[0]:.2f}x+{coeffs_scaled[1]:.2f})')
    plt.plot(x_range_scaled, x_range_scaled, 'g-', linewidth=1, alpha=0.5, label='Perfect prediction')
    plt.xlabel('Predicted (Scaled)', fontsize=12)
    plt.ylabel('Actual (Scaled)', fontsize=12)
    plt.title('Actual vs Predicted Quantity - Transformed Scale (Test Set)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Original Scale ---
    coeffs_orig = np.polyfit(pred_qty, actual_qty, 1)
    fit_orig = np.poly1d(coeffs_orig)
    x_range_orig = np.linspace(pred_qty.min(), pred_qty.max(), 100)

    plt.figure(figsize=(10, 8))
    plt.scatter(pred_qty, actual_qty, alpha=0.5, c='#ff7f0e', edgecolors='none', s=20, label='Test samples')
    plt.plot(x_range_orig, fit_orig(x_range_orig), 'r--', linewidth=2,
             label=f'Best fit (y={coeffs_orig[0]:.2f}x+{coeffs_orig[1]:.2f})')
    plt.plot(x_range_orig, x_range_orig, 'g-', linewidth=1, alpha=0.5, label='Perfect prediction')
    plt.xlabel('Predicted Quantity', fontsize=12)
    plt.ylabel('Actual Quantity', fontsize=12)
    plt.title('Actual vs Predicted Quantity - Original Scale (Test Set)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rmse_curve(train_history, val_history, best_epoch=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label="Training RMSE", linewidth=2.5, color="#1f77b4")
    plt.plot(val_history, label="Validation RMSE", linewidth=2.5, color="#ff7f0e")
    if best_epoch:
        plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best (ep {best_epoch})")
    plt.title("Book Sales Regressor — RMSE", fontsize=18, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_model(X_train, y_train, X_val, y_val, device, verbose=False):
    """Train model and return best RMSE, model state, and training history."""
    model = Regressor(input_dim=X_train.shape[1], dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    train_rmses, val_rmses = [], []
    best_rmse, best_ep = float('inf'), 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item() * len(xb)
            val_loss /= len(X_val)

        # Convert MSE -> RMSE
        train_rmse = float(np.sqrt(train_loss))
        val_rmse = float(np.sqrt(val_loss))
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        # Track best *RMSE*, not MSE
        is_best = val_rmse < best_rmse
        if is_best:
            best_rmse = val_rmse
            best_ep = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if verbose and ((epoch + 1) % 10 == 0 or is_best):
            status = " BEST" if is_best else ""
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Train MSE: {train_loss:.5f} RMSE: {train_rmse:.4f} | "
                f"Val MSE: {val_loss:.5f} RMSE: {val_rmse:.4f}{status}"
            )

    # best_rmse is RMSE, and histories are per-epoch RMSE
    return best_rmse, best_state, train_rmses, val_rmses, best_ep


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    set_seed(SEED)
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

    df_train, df_temp = train_test_split(df_full, test_size=0.2, random_state=SEED, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=SEED, shuffle=True)

    # Store raw data for visualization before transformation
    raw_train = df_train[TRANSFORM_COLS + [TARGET_COL]].copy()

    # 2. Learn Medians (Train only)
    series_medians = df_train.groupby("series")[IMPUTE_COLS].median()
    global_medians = df_train[IMPUTE_COLS].median()

    # 3. Apply Imputation
    print("Imputing missing values...")
    for df in [df_train, df_val, df_test]:
        apply_imputation(df, series_medians, global_medians, IMPUTE_COLS)

    # ------------------------------------------------------------------
    # 4. Transformation Logic
    # ------------------------------------------------------------------
    scalers = {}
    transformed_train = {}

    print("\nTransforming features...")
    for col in TRANSFORM_COLS:
        train_np = df_train[col].to_numpy().astype(np.float32)
        val_np = df_val[col].to_numpy().astype(np.float32)
        test_np = df_test[col].to_numpy().astype(np.float32)

        is_log = col in LOG_COLS
        is_clipped = col in CLIP_COLS

        if is_log:
            train_np = np.log1p(train_np)
            val_np = np.log1p(val_np)
            test_np = np.log1p(test_np)

        scaler = RobustScaler()
        train_np = scaler.fit_transform(train_np.reshape(-1, 1)).flatten()
        val_np = scaler.transform(val_np.reshape(-1, 1)).flatten()
        test_np = scaler.transform(test_np.reshape(-1, 1)).flatten()

        scalers[col] = scaler

        if is_clipped:
            train_np = np.clip(train_np, CLIP_MIN, CLIP_MAX)
            val_np = np.clip(val_np, CLIP_MIN, CLIP_MAX)
            test_np = np.clip(test_np, CLIP_MIN, CLIP_MAX)

        df_train[col] = train_np
        df_val[col] = val_np
        df_test[col] = test_np
        transformed_train[col] = train_np

    # ------------------------------------------------------------------
    # Target Transformation
    # ------------------------------------------------------------------
    print("Fitting target RobustScaler on training data...")

    y_scaler = RobustScaler()
    qty_train_log = np.log1p(df_train[TARGET_COL].to_numpy().reshape(-1, 1))
    y_train_scaled = y_scaler.fit_transform(qty_train_log).flatten()

    y_center = y_scaler.center_[0]
    y_scale = y_scaler.scale_[0]

    transformed_train[TARGET_COL] = y_train_scaled

    print(f"Target transformed: log(quantity) → RobustScaler (median={y_center:.4f}, IQR={y_scale:.4f})")

    # ------------------------------------------------------------------
    # HEATMAP
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
    # SCATTERPLOTS
    # ------------------------------------------------------------------
    scatter_features = ["print_length", "height"]

    print("\nGenerating Scatterplots: Feature vs Quantity...")

    for col in scatter_features:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(raw_train[col], raw_train[TARGET_COL],
                        alpha=0.6, c='#1f77b4', edgecolors='none', s=15)
        axes[0].set_title(f"Raw Reality: {col} vs Quantity", fontsize=14, fontweight='medium')
        axes[0].set_xlabel(f"Raw {col}")
        axes[0].set_ylabel("Raw Quantity")
        axes[0].grid(True, linestyle='--', alpha=0.3)

        axes[1].scatter(df_train[col], transformed_train[TARGET_COL],
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
    df_train_viz = df_train.copy()
    df_train_viz["quantity_raw"] = raw_train[TARGET_COL]
    for cat in DUMMY_COLS:
        (df_train_viz.groupby(["q_num", cat])["quantity_raw"].sum().unstack().fillna(0)
         .plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis", edgecolor="black"))
        plt.title(f"Sales by {cat}")
        plt.tight_layout()
        plt.show()

    # Tensors
    df_train = pd.get_dummies(df_train, columns=DUMMY_COLS, drop_first=True)
    df_val = pd.get_dummies(df_val, columns=DUMMY_COLS, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=DUMMY_COLS, drop_first=True)

    # ------------------------------------------------------------------
    # Phase 1: Greedy Feature Selection (instead of exhaustive search)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1: Greedy Forward Selection")
    print("=" * 60)

    # Prepare y tensors
    qty_val_log = np.log1p(df_val[TARGET_COL].to_numpy().reshape(-1, 1))
    y_val_scaled = y_scaler.transform(qty_val_log).flatten()

    y_train_tensor = torch.from_numpy(y_train_scaled.astype(np.float32)).to(device)
    y_val_tensor = torch.from_numpy(y_val_scaled.astype(np.float32)).to(device)

    best_features, history = greedy_forward_selection(
        POTENTIAL_FEAT_COLS, df_train, df_val, y_train_tensor, y_val_tensor,
        META_COLS, TARGET_COL, device
    )

    # Features to drop = potential - selected
    best_drop_cols = [c for c in POTENTIAL_FEAT_COLS if c not in best_features]
    print(f"\nBest features: {best_features}")
    print(f"Drop: {best_drop_cols}")

    # ------------------------------------------------------------------
    # Phase 2: Retrain with best features from greedy selection
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Retraining BEST combination with full output")
    print("=" * 60)
    print(f"\nBest features: {best_features}")
    print(f"Drop: {best_drop_cols}")
    print("=" * 60)

    # Reset seed and retrain the best combination with verbose output
    set_seed(SEED)

    feat_cols = [c for c in df_train.columns if c not in META_COLS + best_drop_cols + [TARGET_COL]]
    print(f"\nFeatures being used: {feat_cols}")

    X_train = torch.from_numpy(df_train[feat_cols].to_numpy().astype(np.float32)).to(device)
    X_val = torch.from_numpy(df_val[feat_cols].to_numpy().astype(np.float32)).to(device)
    X_test = torch.from_numpy(df_test[feat_cols].to_numpy().astype(np.float32)).to(device)

    qty_test_log = np.log1p(df_test[TARGET_COL].to_numpy().reshape(-1, 1))
    y_test_scaled = y_scaler.transform(qty_test_log).flatten()
    y_test_tensor = torch.from_numpy(y_test_scaled.astype(np.float32)).to(device)

    print("\n=== Dataset Tensor Shapes ===")
    print(f"X_train: {X_train.shape}  →  y_train: {y_train_tensor.shape}")
    print(f"X_val:   {X_val.shape}   →  y_val:   {y_val_tensor.shape}")
    print(f"X_test:  {X_test.shape}  →  y_test:  {y_test_tensor.shape}")
    print(f"Feature dimension used in model → {X_train.shape[1]}")
    print("=" * 34)

    # Train with verbose output
    best_rmse, best_state, train_rmses, val_rmses, best_ep = train_model(
        X_train, y_train_tensor, X_val, y_val_tensor, device, verbose=True
    )

    results_folder = Path("results")
    results_folder.mkdir(parents=True, exist_ok=True)
    # Save best model
    torch.save(best_state, results_folder / "regressor_best.pth")

    # Plot RMSE curve
    plot_rmse_curve(train_rmses, val_rmses, best_ep)

    # ------------------------------------------------------------------
    # Load Best Model & Evaluate on Test Set
    # ------------------------------------------------------------------
    model = Regressor(input_dim=X_train.shape[1], dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load("results/regressor_best.pth"))
    model.eval()

    loss_fn = nn.MSELoss()
    test_dl = DataLoader(TensorDataset(X_test, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in test_dl) / len(X_test)
    test_rmse = np.sqrt(test_loss)

    print(f"\nTest MSE: {test_loss:.5f} RMSE: {test_rmse:.4f}")
    plot_actual_vs_predicted(model, X_test, y_test_tensor, y_scale, y_center, device)

    # ------------------------------------------------------------------
    # Final Prediction
    # ------------------------------------------------------------------
    target_books = pd.read_csv(data_folder / "target_books_new.csv", dtype={"isbn": "string"})
    final_ids = target_books[["isbn", "title"]].copy()

    target_books = apply_imputation(target_books, series_medians, global_medians, IMPUTE_COLS)

    for col in TRANSFORM_COLS:
        vals = target_books[col].to_numpy().astype(np.float32)

        if col in LOG_COLS:
            vals = np.log1p(vals)

        if col in scalers:
            vals = scalers[col].transform(vals.reshape(-1, 1)).flatten()

        if col in CLIP_COLS:
            vals = np.clip(vals, CLIP_MIN, CLIP_MAX)

        target_books[col] = vals

    target_books = pd.get_dummies(target_books, columns=DUMMY_COLS, drop_first=True)
    X_final = torch.from_numpy(
        target_books.reindex(columns=feat_cols, fill_value=0.0).to_numpy().astype(np.float32)
    ).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X_final).cpu().numpy().flatten()

    pred_log = preds * y_scale + y_center
    final_ids["pred_quantity"] = np.expm1(pred_log).round(0).astype(int)

    print("\nFinal Predictions:\n", final_ids.head(10))
    final_ids.to_csv(results_folder / "final_predictions.csv", index=False)
