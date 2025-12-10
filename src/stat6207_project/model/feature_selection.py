from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer  # ← Yeo-Johnson
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
SEED = 42

# Feature Configuration
IMPUTE_COLS = ["length", "width", "rating", "item_weight", "price", "print_length", "height"]
TRANSFORM_COLS = ["q_since_first", "avg_discount_rate", "print_length", "length",
                  "width", "height", "rating", "price", "item_weight"]
CLIP_COLS = ["avg_discount_rate", "rating", "item_weight"]  # Optional clipping after transform
DUMMY_COLS = ["format", "channel", "q_num", "series"]
META_COLS = ["isbn", "title", "year_quarter"]
TARGET_COL = "quantity"
POTENTIAL_FEAT_COLS = [
    "price",
    "height",
    "length",
    "width",
    "item_weight",
    "print_length",
    "rating",
    "q_since_first",
    "avg_discount_rate"
]

# Yeo-Johnson transformers
feature_transformers = {}
target_transformer = None


def set_seed(seed_value=SEED):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)


def apply_imputation(df_target, source_medians, impute_cols):
    df_merged = df_target.join(source_medians, on="series", rsuffix="_median")
    for col in impute_cols:
        median_col = f"{col}_median"
        df_target[col] = df_target[col].fillna(df_merged[median_col])
    return df_target.drop(columns=[c for c in df_target.columns if c.endswith('_median')], errors='ignore')


def transform_features(df, fit=False):
    """Apply Yeo-Johnson transformation (fit on train only)."""
    df = df.copy()
    for col in TRANSFORM_COLS:
        values = df[col].to_numpy(dtype=np.float64).reshape(-1, 1)

        if fit:
            transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            transformed = transformer.fit_transform(values).flatten().astype(np.float32)
            feature_transformers[col] = transformer
        else:
            if col not in feature_transformers:
                raise ValueError(f"Transformer for {col} not fitted!")
            transformed = feature_transformers[col].transform(values).flatten().astype(np.float32)

        if col in CLIP_COLS:
            transformed = np.clip(transformed, -7, 7)

        df[col] = transformed
    return df


def transform_target(values, fit=False):
    """Apply Yeo-Johnson to target."""
    global target_transformer
    values = values.reshape(-1, 1)
    if fit:
        target_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        return target_transformer.fit_transform(values).flatten().astype(np.float32)
    else:
        if target_transformer is None:
            raise ValueError("Target transformer not fitted!")
        return target_transformer.transform(values).flatten().astype(np.float32)


def inverse_transform_target(y_scaled):
    """Inverse Yeo-Johnson back to original quantity."""
    global target_transformer
    if target_transformer is None:
        raise ValueError("Target transformer not fitted!")
    y_scaled = y_scaled.reshape(-1, 1)
    return target_transformer.inverse_transform(y_scaled).flatten()


def greedy_forward_selection(potential_feat_cols, df_train, df_val, y_train_scaled, y_val_scaled,
                             meta_data_cols, target_col, device):
    selected = []
    remaining = list(potential_feat_cols)
    best_rmse_history = []
    all_cols = [c for c in df_train.columns if c not in meta_data_cols + [target_col]]

    print(f"\nStarting Greedy Forward Selection from {len(remaining)} candidates...")

    while remaining:
        best_rmse = float('inf')
        best_feature = None

        for feat in remaining:
            candidate = selected + [feat]
            drop_cols = [c for c in potential_feat_cols if c not in candidate]
            feat_cols = [c for c in all_cols if c not in drop_cols]

            X_train = torch.from_numpy(df_train[feat_cols].to_numpy(np.float32)).to(device)
            X_val = torch.from_numpy(df_val[feat_cols].to_numpy(np.float32)).to(device)

            set_seed(SEED)
            rmse, _, _, _, _ = train_model(X_train, y_train_scaled, X_val, y_val_scaled, device, verbose=False)

            if rmse < best_rmse:
                best_rmse = rmse
                best_feature = feat

        if best_feature is None or best_rmse >= min([h[1] for h in best_rmse_history], default=float('inf')):
            print("No improvement. Stopping.")
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_rmse_history.append((selected.copy(), best_rmse))
        print(f"Added: {best_feature:<15} → Val RMSE: {best_rmse:.4f} | Features: {len(selected)}")

    best_features = min(best_rmse_history, key=lambda x: x[1])[0]
    return best_features, best_rmse_history


def train_model(X_train, y_train, X_val, y_val, device, verbose=False):
    model = Regressor(input_dim=X_train.shape[1], dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    best_rmse = float('inf')
    best_state = None
    train_rmses, val_rmses = [], []

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
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item() * xb.size(0)
        val_loss /= len(val_dl.dataset)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())

        if verbose and ((epoch + 1) % 20 == 0 or val_rmse < best_rmse):
            status = " [BEST]" if val_rmse < best_rmse else ""
            print(f"Ep {epoch+1:3d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}{status}")

    return best_rmse, best_state, train_rmses, val_rmses, np.argmin(val_rmses) + 1


def plot_actual_vs_predicted(y_actual_orig, y_pred_orig, data_type="Test"):
    plt.figure(figsize=(10, 8))
    y_pred_clip = np.clip(y_pred_orig, 0, None)
    plt.scatter(y_pred_clip, y_actual_orig, alpha=0.6, c='#1f77b4', s=30, edgecolor='none')
    min_val, max_val = min(y_actual_orig.min(), y_pred_clip.min()), max(y_actual_orig.max(), y_pred_clip.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', lw=2, label='Perfect')
    coeffs = np.polyfit(y_pred_clip, y_actual_orig, 1)
    plt.plot(np.linspace(min_val, max_val), np.poly1d(coeffs)(np.linspace(min_val, max_val)), 'r-', label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.1f}')
    plt.xlabel("Predicted Quantity")
    plt.ylabel("Actual Quantity")
    plt.title(f"Actual vs Predicted ({data_type} Set - Original Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    data_folder = Path("data")
    df_full = pd.read_csv(data_folder / "target_series_new_with_features2.csv", dtype={"isbn": "string"})

    df_train, df_temp = train_test_split(df_full, test_size=0.2, random_state=SEED, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=SEED, shuffle=True)

    # 1. Imputation
    series_medians = df_train.groupby("series")[IMPUTE_COLS].median()
    for df in [df_train, df_val, df_test]:
        apply_imputation(df, series_medians, IMPUTE_COLS)

    # 2. Fit Yeo-Johnson on train
    print("\nFitting Yeo-Johnson transformers on training data...")
    df_train = transform_features(df_train, fit=True)
    df_val = transform_features(df_val, fit=False)
    df_test = transform_features(df_test, fit=False)

    y_train_scaled = transform_target(df_train[TARGET_COL].values, fit=True)
    y_val_scaled = transform_target(df_val[TARGET_COL].values, fit=False)
    y_test_scaled = transform_target(df_test[TARGET_COL].values, fit=False)

    print(f"Target Yeo-Johnson λ = {target_transformer.lambdas_[0]:.4f}")

    # 3. Dummies
    df_train = pd.get_dummies(df_train, columns=DUMMY_COLS, drop_first=True)
    df_val = pd.get_dummies(df_val, columns=DUMMY_COLS, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=DUMMY_COLS, drop_first=True)

    # Align columns
    df_val = df_val.reindex(columns=df_train.columns, fill_value=0)
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
    df_val = df_val.drop(columns=[TARGET_COL], errors='ignore')
    df_test = df_test.drop(columns=[TARGET_COL], errors='ignore')

    # 4. Greedy Forward Selection
    y_train_tensor = torch.from_numpy(y_train_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val_scaled).to(device)

    best_features, history = greedy_forward_selection(
        POTENTIAL_FEAT_COLS, df_train, df_val, y_train_tensor, y_val_tensor,
        META_COLS, TARGET_COL, device
    )

    print(f"\nBest feature set: {best_features}")

    # 5. Final Training
    drop_cols = [c for c in POTENTIAL_FEAT_COLS if c not in best_features]
    final_cols = [c for c in df_train.columns if c not in META_COLS + drop_cols + [TARGET_COL]]

    X_train = torch.from_numpy(df_train[final_cols].to_numpy(np.float32)).to(device)
    X_val = torch.from_numpy(df_val[final_cols].to_numpy(np.float32)).to(device)
    X_test = torch.from_numpy(df_test[final_cols].to_numpy(np.float32)).to(device)
    y_test_tensor = torch.from_numpy(y_test_scaled).to(device)

    print(f"\nFinal training with {len(final_cols)} features...")
    best_rmse, best_state, train_hist, val_hist, best_ep = train_model(
        X_train, y_train_tensor, X_val, y_val_tensor, device, verbose=True
    )

    # Save model
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True, parents=True)
    torch.save(best_state, results_folder / "regressor_best_yeojohnson.pth")

    # Evaluate
    model = Regressor(input_dim=X_test.shape[1], dropout=DROPOUT).to(device)
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_pred_scaled = model(X_test).cpu().numpy().flatten()
    test_rmse_scaled = np.sqrt(nn.MSELoss()(torch.from_numpy(test_pred_scaled), y_test_tensor.cpu()).item())
    print(f"\nTest RMSE (scaled space): {test_rmse_scaled:.4f}")

    # Inverse transform for plotting
    y_test_orig = inverse_transform_target(y_test_scaled)
    y_pred_orig = inverse_transform_target(test_pred_scaled)
    plot_actual_vs_predicted(y_test_orig, y_pred_orig, "Test")

    # Final predictions
    target_books = pd.read_csv(data_folder / "target_books_new.csv", dtype={"isbn": "string"})
    ids = target_books[["isbn", "title"]].copy()

    target_books = apply_imputation(target_books, series_medians, IMPUTE_COLS)
    target_books = transform_features(target_books, fit=False)
    target_books = pd.get_dummies(target_books, columns=DUMMY_COLS, drop_first=True)
    target_books = target_books.reindex(columns=final_cols, fill_value=0.0)

    X_final = torch.from_numpy(target_books.to_numpy(np.float32)).to(device)
    with torch.no_grad():
        pred_scaled = model(X_final).cpu().numpy().flatten()
    pred_qty = np.clip(inverse_transform_target(pred_scaled), 0, None).round().astype(int)

    ids["pred_quantity"] = pred_qty
    ids.to_csv(results_folder / "final_predictions_yeojohnson.csv", index=False)
    print("\nFinal predictions saved!")
    print(ids.head(10))