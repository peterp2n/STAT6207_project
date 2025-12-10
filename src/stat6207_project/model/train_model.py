# train_model_yeojohnson_safe.py
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from regressor import Regressor
import copy

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SEED = 42
DROPOUT = 0.0
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 150
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Columns
IMPUTE_COLS = ["length", "width", "rating", "item_weight", "price", "print_length", "height"]
TRANSFORM_COLS = ["q_since_first", "avg_discount_rate", "print_length", "length",
                  "width", "height", "rating", "price", "item_weight"]
CLIP_COLS = ["avg_discount_rate", "rating", "item_weight"]
DUMMY_COLS = ["format", "channel", "q_num", "series"]
META_COLS = ["isbn", "title", "year_quarter"]
TARGET_COL = "quantity"

# Global transformers (fitted once on train)
feature_transformers = {}
target_transformer = None


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_imputation(df, series_medians):
    df = df.copy()
    df_merged = df.join(series_medians, on="series", rsuffix="_median")
    for col in IMPUTE_COLS:
        median_col = f"{col}_median"
        df[col] = df[col].fillna(df_merged[median_col])
    return df.drop(columns=[c for c in df.columns if c.endswith('_median')], errors='ignore')


def fit_transform_features(df):
    df = df.copy()
    for col in TRANSFORM_COLS:
        x = df[col].to_numpy(dtype=np.float64).reshape(-1, 1)
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        transformed = pt.fit_transform(x).flatten().astype(np.float32)
        feature_transformers[col] = pt
        if col in CLIP_COLS:
            transformed = np.clip(transformed, -7, 7)
        df[col] = transformed
    return df


def transform_features(df):
    df = df.copy()
    for col in TRANSFORM_COLS:
        x = df[col].to_numpy(dtype=np.float64).reshape(-1, 1)
        transformed = feature_transformers[col].transform(x).flatten().astype(np.float32)
        if col in CLIP_COLS:
            transformed = np.clip(transformed, -7, 7)
        df[col] = transformed
    return df


def fit_transform_target(y):
    global target_transformer
    y = y.reshape(-1, 1)
    target_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    return target_transformer.fit_transform(y).flatten().astype(np.float32)


def transform_target(y):
    global target_transformer
    y = y.reshape(-1, 1)
    return target_transformer.transform(y).flatten().astype(np.float32)


def inverse_target(y_scaled):
    global target_transformer
    y_scaled = y_scaled.reshape(-1, 1)
    return target_transformer.inverse_transform(y_scaled).flatten()


def safe_get_dummies(train_df, val_df, test_df):
    """
    Fit get_dummies on train only → apply same columns to val/test.
    Prevents KeyError and shape mismatch.
    """
    train_dummy = pd.get_dummies(train_df, columns=DUMMY_COLS, drop_first=True, dtype=int)
    val_dummy = pd.get_dummies(val_df, columns=DUMMY_COLS, drop_first=True, dtype=int)
    test_dummy = pd.get_dummies(test_df, columns=DUMMY_COLS, drop_first=True, dtype=int)

    # Use train columns as reference
    dummy_cols = [c for c in train_dummy.columns if any(c.startswith(prefix + "_") for prefix in DUMMY_COLS)]

    # Reindex val/test to match train exactly
    val_dummy = val_dummy.reindex(columns=train_dummy.columns, fill_value=0)
    test_dummy = test_dummy.reindex(columns=train_dummy.columns, fill_value=0)

    # Drop target and meta if present
    for df in [train_dummy, val_dummy, test_dummy]:
        df.drop(columns=[TARGET_COL] + META_COLS, errors='ignore', inplace=True)

    print(f"Dummy variables aligned: {len(dummy_cols)} columns from {DUMMY_COLS}")
    return train_dummy, val_dummy, test_dummy, dummy_cols


def train_model_final(X_train, y_train, X_val, y_val, X_test, y_test, device):
    model = Regressor(input_dim=X_train.shape[1], dropout=DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    best_rmse = float('inf')
    best_state = None
    history = {"train": [], "val": []}

    print(f"\nTraining final model → {X_train.shape[1]} features | {EPOCHS} epochs")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= len(val_dl.dataset)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        history["train"].append(train_rmse)
        history["val"].append(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_ep = epoch

        if epoch % 20 == 0 or val_rmse < best_rmse:
            print(f"Ep {epoch:3d} → Train: {train_rmse:.4f} | Val: {val_rmse:.4f} {'[BEST]' if val_rmse < best_rmse else ''}")

    print(f"\nBest Val RMSE: {best_rmse:.4f} at epoch {best_ep}")
    return best_state, history, best_rmse


if __name__ == "__main__":
    set_seed()
    print(f"Using device: {DEVICE}")

    # Load data
    df = pd.read_csv("data/target_series_new_with_features2.csv", dtype={"isbn": "string"})

    # Split first (before any processing)
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=SEED, shuffle=True)

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # 1. Imputation (fit on train)
    series_medians = df_train.groupby("series")[IMPUTE_COLS].median()
    global_medians = df_train[IMPUTE_COLS].median()
    df_train = apply_imputation(df_train, series_medians)
    df_val   = apply_imputation(df_val,   series_medians)
    df_test  = apply_imputation(df_test,  series_medians)

    # 2. Yeo-Johnson (fit on train only)
    df_train = fit_transform_features(df_train)
    df_val   = transform_features(df_val)
    df_test  = transform_features(df_test)

    y_train_scaled = fit_transform_target(df_train[TARGET_COL].values)
    y_val_scaled   = transform_target(df_val[TARGET_COL].values)
    y_test_scaled  = transform_target(df_test[TARGET_COL].values)

    print(f"Target λ = {target_transformer.lambdas_[0]:.4f}")

    # 3. SAFE DUMMY ENCODING
    df_train, df_val, df_test, dummy_cols = safe_get_dummies(df_train, df_val, df_test)

    # 4. Final feature selection (example: use all transformed + dummies)
    feature_cols = [c for c in df_train.columns if c not in META_COLS + [TARGET_COL]]

    X_train = torch.from_numpy(df_train[feature_cols].to_numpy(np.float32)).to(DEVICE)
    X_val   = torch.from_numpy(df_val[feature_cols].to_numpy(np.float32)).to(DEVICE)
    X_test  = torch.from_numpy(df_test[feature_cols].to_numpy(np.float32)).to(DEVICE)

    y_train = torch.from_numpy(y_train_scaled).to(DEVICE)
    y_val   = torch.from_numpy(y_val_scaled).to(DEVICE)
    y_test  = torch.from_numpy(y_test_scaled).to(DEVICE)

    print(f"Final feature count: {len(feature_cols)} (including {len(dummy_cols)} dummies)")

    # 5. Train final model
    best_state, history, best_rmse = train_model_final(X_train, y_train, X_val, y_val, X_test, y_test, DEVICE)

    # Save
    Path("results").mkdir(exist_ok=True)
    # torch.save(best_state, "results/regressor_yeojohnson_final.pth")

    # Test evaluation in original scale
    model = Regressor(input_dim=X_test.shape[1], dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test).cpu().numpy().flatten()

    y_test_orig = inverse_target(y_test_scaled)
    y_pred_orig = inverse_target(pred_scaled)

    print(f"\nTest RMSE (original scale): {np.sqrt(np.mean((y_test_orig - y_pred_orig)**2)):.2f}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred_orig, y_test_orig, alpha=0.6, s=30)
    plt.plot([0, y_test_orig.max()], [0, y_test_orig.max()], 'r--')
    plt.xlabel("Predicted Quantity")
    plt.ylabel("Actual Quantity")
    plt.title("Final Model - Actual vs Predicted (Yeo-Johnson + Safe Dummies)")
    plt.grid(alpha=0.3)
    plt.show()