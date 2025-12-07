from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from regressor import Regressor

device = torch.device("mps")


def apply_imputation(df_target, source_medians, fallback_medians, impute_cols):
    """
    Applies series-specific medians for selected columns.
    """
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

    # ------------------------------------------------------------------
    # STEVE JOBS FIX: Capture the "Missingness" signal BEFORE we polish
    # ------------------------------------------------------------------
    # We explicitly tell the model: "We don't know the length of this one."
    # This allows us to use the Median for the value (neutralizing the flip)
    # while still letting the model learn that "Unknown Length" is a special category.
    flag_cols = ["print_length", "height"]
    for col in flag_cols:
        df_full[f"{col}_is_missing"] = df_full[col].isna().astype(np.float32)

    df_train, df_temp = train_test_split(df_full, test_size=0.3, random_state=42, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

    # ------------------------------------------------------------------
    # 1. Configuration: Imputation Split
    # ------------------------------------------------------------------
    # We moved print_length and height BACK to impute_cols.
    # Why? Because Median (Average) is "safe". It sits in the middle.
    # It does not pull the regression line up or down. It stops the flip.
    impute_cols = ["length", "width", "rating", "item_weight", "price", "print_length", "height"]

    transform_cols = ["q_since_first", "avg_discount_rate", "print_length", "length", "width", "height", "rating",
                      "price"]
    raw_train = df_train[transform_cols + ["quantity"]].copy()  # Capture for Before Heatmap/Boxplots

    # 2. Learn Medians (Train only)
    series_medians = df_train.groupby("series")[impute_cols].median()
    global_medians = df_train[impute_cols].median()

    # 3. Apply Imputation (Median for ALL, No more Zero filling)
    print("Imputing missing values...")
    for df in [df_train, df_val, df_test]:
        apply_imputation(df, series_medians, global_medians, impute_cols)

    # ------------------------------------------------------------------
    # 4. Transformation Logic
    # ------------------------------------------------------------------
    log_cols = ["print_length", "length", "width", "height", "price", "q_since_first"]
    train_stats = {}
    transformed_train = {}

    print("\nTransforming features...")
    for col in transform_cols:
        train_np = df_train[col].to_numpy().astype(np.float32)
        val_np = df_val[col].to_numpy().astype(np.float32)
        test_np = df_test[col].to_numpy().astype(np.float32)

        is_log = col in log_cols
        if is_log:
            train_np = np.log1p(train_np)
            val_np = np.log1p(val_np)
            test_np = np.log1p(test_np)

        _mean = train_np.mean()
        _std = train_np.std() + 1e-8
        train_stats[col] = {"mean": _mean, "std": _std, "log": is_log}

        # Standardize & Clip
        train_np = np.clip((train_np - _mean) / _std, -3, 3)
        val_np = np.clip((val_np - _mean) / _std, -3, 3)
        test_np = np.clip((test_np - _mean) / _std, -3, 3)

        df_train[col] = train_np
        df_val[col] = val_np
        df_test[col] = test_np
        transformed_train[col] = train_np

    # ------------------------------------------------------------------
    # 5. Visual Validation: Boxplots
    # ------------------------------------------------------------------
    print("\nGenerating Side-by-Side Boxplots...")
    for col in transform_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)  # Different scales

        # Plot Raw (original data, NaNs handled automatically by plot)
        raw_train[col].plot.box(ax=axes[0])
        axes[0].set_title(f"Raw {col} (Train Set)")
        axes[0].set_ylabel(col)

        # Plot Processed (Imputed + Transformed)
        df_train[col].plot.box(ax=axes[1])
        axes[1].set_title(f"Preprocessed {col} (Train Set)")
        axes[1].set_ylabel("Transformed Value (Std)")

        plt.suptitle(f"Preprocessing Effect: {col}", fontsize=14)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # HEATMAP
    # ------------------------------------------------------------------
    qty_raw = raw_train["quantity"].to_numpy().astype(np.float32)
    qty_trans = np.clip((np.log1p(qty_raw) - np.log1p(qty_raw).mean()) / np.log1p(qty_raw).std(), -3, 3)
    transformed_train["quantity"] = qty_trans

    corr_before = raw_train.corr()
    corr_after = pd.DataFrame(transformed_train).corr()

    fig, axes = plt.subplots(1, 2, figsize=(32, 14))
    heatmap_args = {"annot": True, "fmt": ".2f", "cmap": "coolwarm", "center": 0, "square": True}

    sns.heatmap(corr_before, ax=axes[0], **heatmap_args)
    axes[0].set_title("Correlation: Raw Train Data (Before)", fontsize=18)

    sns.heatmap(corr_after, ax=axes[1], **heatmap_args)
    axes[1].set_title("Correlation: Transformed Train Data (After)", fontsize=18)
    plt.tight_layout();
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
    feat_cols = [c for c in df_train.columns if c not in ["isbn", "title", "year_quarter", "series", target_col]]

    # Note: feat_cols AUTOMATICALLY includes our new "print_length_is_missing" flags
    # because they are in df_train and not in the excluded list.
    print(f"Features being used: {feat_cols}")

    X_train = torch.from_numpy(df_train[feat_cols].to_numpy().astype(np.float32)).to(device)
    X_val = torch.from_numpy(df_val[feat_cols].to_numpy().astype(np.float32)).to(device)
    X_test = torch.from_numpy(df_test[feat_cols].to_numpy().astype(np.float32)).to(device)

    # Target Transform
    qty_log_train = np.log1p(df_train[target_col])
    target_mean, target_std = qty_log_train.mean(), qty_log_train.std()

    y_train = torch.from_numpy(((qty_log_train - target_mean) / target_std).to_numpy().astype(np.float32)).to(device)
    y_val = torch.from_numpy(
        ((np.log1p(df_val[target_col]) - target_mean) / target_std).to_numpy().astype(np.float32)).to(device)
    y_test = torch.from_numpy(
        ((np.log1p(df_test[target_col]) - target_mean) / target_std).to_numpy().astype(np.float32)).to(device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    model = Regressor(input_dim=X_train.shape[1], dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)

    train_rmses, val_rmses, best_rmse, best_ep = [], [], float('inf'), 0

    for epoch in range(50):
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
    # Final Prediction
    # ------------------------------------------------------------------
    target_books = pd.read_csv(data_folder / "target_books_new.csv", dtype={"isbn": "string"})
    final_ids = target_books[["isbn", "title"]].copy()

    # 1. APPLY THE SAME FLAG LOGIC
    for col in flag_cols:
        target_books[f"{col}_is_missing"] = target_books[col].isna().astype(np.float32)

    # 2. Impute (Median where valid)
    target_books = apply_imputation(target_books, series_medians, global_medians, impute_cols)

    # 3. Transform
    for col in transform_cols:
        s = train_stats[col]
        vals = target_books[col].to_numpy().astype(np.float32)
        if s["log"]: vals = np.log1p(vals)
        target_books[col] = np.clip((vals - s["mean"]) / s["std"], -3, 3)

    # 4. Predict
    target_books = pd.get_dummies(target_books, columns=dummy_cols, drop_first=True)
    X_final = torch.from_numpy(target_books.reindex(columns=feat_cols, fill_value=0.0).values.astype(np.float32)).to(
        device)

    model.eval()
    with torch.no_grad():
        preds = model(X_final).cpu().numpy().flatten()

    final_ids["pred_quantity"] = np.expm1(preds * target_std + target_mean).round(0).astype(int)
    print("\nFinal Predictions:\n", final_ids.head(10))
    final_ids.to_csv("results/final_predictions.csv", index=False)
