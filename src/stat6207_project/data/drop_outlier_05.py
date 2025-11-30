from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from src.stat6207_project.data.preprocess_util_04 import preprocess_data, to_tensors

# Nice default style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_outlier_comparison_matplotlib(
        X_full: np.ndarray,
        outlier_mask: np.ndarray,
        X_clean: np.ndarray,
        *,
        perplexity: int = 40,
        random_state: int = 42,
        marker_size: int = 8,
        outlier_marker_size: int = 35,
        figsize: tuple = (15, 6),
        save_path: str | None = "tsne_outlier_removal.png",
) -> None:
    # Force boolean mask
    if outlier_mask.dtype != bool:
        is_outlier = outlier_mask == -1
    else:
        is_outlier = outlier_mask

    n_total = len(X_full)
    n_outliers = int(is_outlier.sum())
    n_clean = len(X_clean)

    # CRITICAL FIX: Force NumPy arrays — this stops Polars interference forever
    tsne_full = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    X_tsne_full = np.asarray(tsne_full.fit_transform(X_full))  # ← THIS LINE FIXES IT

    tsne_clean = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    X_tsne_clean = np.asarray(tsne_clean.fit_transform(X_clean))  # ← AND THIS ONE

    # Now safe to use boolean indexing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Before
    ax1.scatter(X_tsne_full[~is_outlier, 0], X_tsne_full[~is_outlier, 1],
                c="steelblue", s=marker_size, alpha=0.8, label="Inliers")
    ax1.scatter(X_tsne_full[is_outlier, 0], X_tsne_full[is_outlier, 1],
                c="crimson", s=outlier_marker_size, edgecolor="black", linewidth=0.8,
                label="Outliers", alpha=0.9)
    ax1.set_title(f"Before (n={n_total:,}, {n_outliers:,} outliers)")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.legend()

    # After
    ax2.scatter(X_tsne_clean[:, 0], X_tsne_clean[:, 1],
                c="steelblue", s=marker_size, alpha=0.8)
    ax2.set_title(f"After (n={n_clean:,} clean)")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    fig.suptitle("t-SNE: Before vs After Outlier Removal", fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

    print(f"Removed {n_outliers:,} outliers ({n_outliers / n_total:.2%})")


def get_cleaned_dataframes(df_original: pl.DataFrame, inlier_mask: np.ndarray) -> pl.DataFrame:
    """
    Returns the cleaned version of the original DataFrame with outliers removed.

    Args:
        df_original (pl.DataFrame): The original DataFrame before outlier removal.
        inlier_mask (np.ndarray): Boolean array where True indicates inliers.

    Returns:
        pl.DataFrame: Cleaned DataFrame with only inliers.
    """
    if len(inlier_mask) != len(df_original):
        raise ValueError("Inlier mask length must match the DataFrame length.")
    return df_original.filter(pl.Series(inlier_mask))


def get_cleaned_dataframes(df_original: pl.DataFrame, inlier_mask: np.ndarray) -> pl.DataFrame:
    """
    Returns the cleaned version of the original DataFrame with outliers removed.

    Args:
        df_original (pl.DataFrame): The original DataFrame before outlier removal.
        inlier_mask (np.ndarray): Boolean array where True indicates inliers.

    Returns:
        pl.DataFrame: Cleaned DataFrame with only inliers.
    """
    if len(inlier_mask) != len(df_original):
        raise ValueError("Inlier mask length must match the DataFrame length.")
    return df_original.filter(pl.Series(inlier_mask))


if __name__ == "__main__":
    # --------------------------- Load data ---------------------------------
    df_books = pl.read_csv("data/merged3.csv", schema_overrides={"isbn": pl.Utf8})
    df_sales = (
        pl.read_csv(
            "expand_quarterly_sales_retail.csv",
            schema_overrides={"barcode2": pl.Utf8, "Quarter_num": pl.Utf8},
        )
        .rename({"barcode2": "isbn"})
    )

    unwanted_cols = [
        "isbn", "title",  # Removed 'author' if not present
        "Next_Q1", "Next_Q2", "Next_Q3", "Next_Q4", "Next_Q2_log1p", "Next_Q3_log1p", "Next_Q4_log1p",
        "Current_quarter", "First_day", "Avg_discount", "print_length", "item_weight", "length", "width", "height",
        "rating", "number_of_reviews", "customer_reviews", "best_sellers_rank", "price",
        "Previous_quarter_qty", "Current_quarter_qty"
    ]

    target_col = "Next_Q1_log1p"

    # --------------------------- Preprocess --------------------------------
    X_train, X_test, _, _ = preprocess_data(books_df=df_books, sales_df=df_sales, test_size=0.2, random_state=42, shuffle=True)

    # Keep only the features we will actually train on
    feats = [c for c in X_train.columns if c != target_col and c not in unwanted_cols]

    X_train_np = X_train.select(feats).to_numpy().astype(np.float32)
    X_test_np = X_test.select(feats).to_numpy().astype(np.float32)

    # Check for NaNs/infs (add this for robustness)
    if np.any(np.isnan(X_train_np)) or np.any(np.isinf(X_train_np)):
        raise ValueError("NaNs or Infs detected in training features – handle in preprocess_data.")

    print(f"Feature matrix shape: {X_train_np.shape}")

    # --------------------------- t-SNE + DBSCAN ----------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        init="pca",
        learning_rate="auto",
        random_state=42,
        n_jobs=-1,
    )
    X_2d = tsne.fit_transform(X_train_np)

    db = DBSCAN(eps=3.0, min_samples=10)  # Tune eps/min_samples based on plot
    labels = db.fit_predict(X_2d)  # -1 = outlier
    inlier_mask = labels != -1  # True = keep

    print(f"DBSCAN found {(labels == -1).sum():,} outliers")

    # --------------------------- Visualization ------------------------------
    plot_outlier_comparison_matplotlib(
        X_full=X_train_np,
        outlier_mask=labels == -1,
        X_clean=X_train_np[inlier_mask],
        perplexity=40,
        save_path="tsne_outlier_removal_before_after.png",
    )

    # --------------------------- Get Cleaned DataFrames --------------------
    X_train_clean = get_cleaned_dataframes(X_train, inlier_mask)
    print(f"Cleaned training DataFrame shape: {X_train_clean.shape}")

    # --------------------------- Convert to PyTorch tensors ----------------
    X_tensor, y_tensor, _ = to_tensors(
        df=X_train_clean,
        target_col=target_col,
        drop_cols=unwanted_cols,
    )

    X_numpy = X_tensor.numpy()
    y_numpy = y_tensor.numpy()

    save_dir = Path("data")
    # Save .npy (pure features/target numerics)
    np.save(save_dir / "X_train.npy", X_numpy)
    np.save(save_dir / "y_train.npy", y_numpy)
    np.save(save_dir / "X_test.npy", X_test_np)
    np.save(save_dir / "y_test.npy", X_test.select(target_col).to_numpy().astype(np.float32))

    # Save CSVs with ISBN (features separate from target)

    X_train_clean.select(["isbn"] + feats).write_csv(save_dir / "X_train_with_isbn.csv")  # ISBN + features
    X_train_clean.select(["isbn", target_col]).write_csv(save_dir / "y_train_with_isbn.csv")  # ISBN + target
    X_test.select(["isbn"] + feats).write_csv(save_dir / "X_test_with_isbn.csv")
    X_test.select(["isbn", target_col]).write_csv(save_dir / "y_test_with_isbn.csv")

    print("\nReady for training!")
    print(f"   X_tensor shape : {X_tensor.shape}")
    print(f"   y_tensor shape : {y_tensor.shape}")