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

    series_path = data_folder / "target_series_new_with_features2.csv"
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

    # Capture raw train data before any imputation or transformation, including quantity for correlations
    transform_cols = [
        "q_since_first", "avg_discount_rate", "print_length", "length", "width", "height", "rating", "price"
    ]
    continuous_cols = transform_cols + ["quantity"]
    raw_train = df_train[continuous_cols].copy()

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
    transformed_train = {}  # Collect transformed features for "after" correlation
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

        transformed_train[col] = train_np  # Store transformed train features

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

    # New: Transform quantity for "after" correlation (using train stats, but only for viz)
    qty_train_log = np.log1p(df_train["quantity"])
    qty_train_mean = qty_train_log.mean()
    qty_train_std = qty_train_log.std()
    qty_train_clipped = np.clip(qty_train_log, qty_train_mean - 3 * qty_train_std, qty_train_mean + 3 * qty_train_std)
    qty_train_transformed = (qty_train_clipped - qty_train_clipped.mean()) / qty_train_clipped.std()
    transformed_train["quantity"] = qty_train_transformed

    # Compute correlations
    corr_before = raw_train.corr()  # Pairwise deletion for NaNs
    transformed_df = pd.DataFrame(transformed_train)
    corr_after = transformed_df.corr()

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(32, 14))

    heatmap_args = {
        "annot": True,
        "fmt": ".2f",
        "cmap": "coolwarm",
        "center": 0,
        "square": True,
        "linewidths": 1,
        "cbar_kws": {"shrink": 0.7},
        "annot_kws": {"size": 11, "weight": "bold"}
    }

    sns.heatmap(corr_before, ax=axes[0], **heatmap_args)
    axes[0].set_title("Correlation: Raw Train Data (Before)\n(Pair-wise Deletion)", fontsize=18, pad=20)
    axes[0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0].tick_params(axis='y', rotation=0, labelsize=12)

    sns.heatmap(corr_after, ax=axes[1], **heatmap_args)
    axes[1].set_title("Correlation: Transformed Train Data (After)\n(Imputed + Log1p + Clipped + Std)", fontsize=18,
                      pad=20)
    axes[1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1].tick_params(axis='y', rotation=0, labelsize=12)

    plt.tight_layout()
    plt.show()

    # ===================================================================
    # NEW: Stacked bar charts – Total sales by quartile × format / channel
    # ===================================================================
    print("\nPlotting sales distribution across quarters...")

    # Make sure we are working with the *original* (non-transformed) quantity
    # because we want real-world book counts on the y-axis
    df_train_for_viz = df_train.copy()
    df_train_for_viz["quantity_raw"] = raw_train["quantity"]  # this is the untouched target

    categorical_cols = ["format", "channel"]

    for cat_col in categorical_cols:
        # Aggregate total quantity sold per (q_num, category)
        agg = (
            df_train_for_viz
            .groupby(["q_num", cat_col], as_index=False)["quantity_raw"]
            .sum()
            .rename(columns={"quantity_raw": "total_quantity"})
        )

        # Pivot so each category becomes a column → ready for stacked bar
        pivot = agg.pivot(index="q_num", columns=cat_col, values="total_quantity").fillna(0)

        # Plot
        ax = pivot.plot(kind="bar",
                        stacked=True,
                        figsize=(10, 6),
                        colormap="viridis",
                        edgecolor="black",
                        linewidth=0.5)

        plt.title(f"Total Books Sold by Quarter — Stacked by {cat_col.capitalize()}",
                  fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Quarter (q_num)", fontsize=12)
        plt.ylabel("Total Quantity Sold", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title=cat_col.capitalize(), bbox_to_anchor=(1.05, 1), loc="upper left")
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


    dummy_cols = ["format", "channel"]
    df_train = pd.get_dummies(df_train, columns=dummy_cols, drop_first=True)
    df_val = pd.get_dummies(df_val, columns=dummy_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=dummy_cols, drop_first=True)

    feat_cols = [c for c in df_train.columns if c not in (metadata_cols + [target_col])]

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

    # ===================================================================
    # FINAL PREDICTION — Perfect, safe, and preserves book identity
    # ===================================================================

    target_books_new = pd.read_csv(data_folder / "target_books_new.csv", dtype={"isbn": "string"})

    # Preserve identity columns before anything touches them
    identity_cols = target_books_new[["isbn", "title"]].copy()

    # 1. Imputation
    target_books_new = apply_imputation(target_books_new.copy(), series_medians, global_medians)

    # 2. Numerical transformations (using the SAME stats from training)
    # Note: We save these stats properly now
    transform_stats = {}
    for col in transform_cols:
        x = np.log1p(target_books_new[col])
        mean = df_train[col].mean()  # already transformed in training loop
        std = df_train[col].std()
        x = (x - mean) / std
        x = np.clip(x, -3, 3)
        target_books_new[col] = x
        transform_stats[col] = (mean, std)

    # 3. Dummies
    dummy_cols = ["format", "channel", "q_num"]
    target_books_new = pd.get_dummies(target_books_new, columns=dummy_cols, drop_first=True)

    # 4. Align features EXACTLY with training — but keep identity separate
    X_target = target_books_new.reindex(columns=feat_cols, fill_value=0.0)

    # Now predict
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_target.values.astype(np.float32)).to(device)
        preds = model(X_tensor).cpu().numpy().flatten()

    # Reverse transform
    pred_quantity = np.expm1(preds * target_train_std + target_train_mean)

    # Reattach identity
    result = identity_cols.copy()
    result["pred_quantity"] = pred_quantity.round(0).astype(int)  # real book counts

    print("Predictions (final, clean, beautiful):")
    print(result.head(10))
    result.to_csv("results/final_predictions.csv", index=False)
    print("\nPredictions saved to results/final_predictions.csv")
    print("end")