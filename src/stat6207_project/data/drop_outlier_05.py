import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from pathlib import Path
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from src.stat6207_project.data.preprocess_util_04 import preprocess_data, to_tensors

def plot_outlier_comparison(
    X_full: np.ndarray,
    outlier_mask: np.ndarray,        # True = outlier, False = inlier   (or use labels == -1)
    X_clean: np.ndarray,
    *,
    perplexity: int = 40,
    random_state: int = 42,
    marker_size: int = 5,
    height: int = 550,
    width: int = 1300,
) -> None:
    """
    Side-by-side t-SNE visualisation before / after outlier removal.

    Parameters
    ----------
    X_full : np.ndarray
        Original feature matrix (already imputed / scaled) – shape (n_samples, n_features)
    outlier_mask : np.ndarray (bool) or (int)
        Boolean mask where True = outlier (or DBSCAN labels where -1 = outlier)
    X_clean : np.ndarray
        Feature matrix after removing the outliers have been dropped
    perplexity, random_state : passed to TSNE
    """

    # ------------------------------------------------------------------ #
    # 1. t-SNE on the full data (so we can colour the outliers)
    # ------------------------------------------------------------------ #
    tsne_full = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    X_tsne_full = tsne_full.fit_transform(X_full)

    # ------------------------------------------------------------------ #
    # 2. t-SNE on the cleaned data
    # ------------------------------------------------------------------ #
    tsne_clean = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    X_tsne_clean = tsne_clean.fit_transform(X_clean)

    # ------------------------------------------------------------------ #
    # 3. Prepare colours for the "before" plot
    # ------------------------------------------------------------------ #
    # Accept both a boolean mask (True=outlier) and DBSCAN labels (-1=outlier)
    if outlier_mask.dtype == bool:
        is_outlier = outlier_mask
    else:
        is_outlier = outlier_mask == -1

    colors_before = np.where(is_outlier, "crimson", "steelblue")

    # ------------------------------------------------------------------ #
    # 4. Build Plotly figure
    # ------------------------------------------------------------------ #
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Before – {X_full.shape[0]:,} samples "
            f"({is_outlier.sum():,} outliers highlighted)",
            f"After – {X_clean.shape[0]:,} samples (clean)",
        ),
        horizontal_spacing=0.12,
    )

    # Left – before
    fig.add_trace(
        go.Scatter(
            x=X_tsne_full[:, 0],
            y=X_tsne_full[:, 1],
            mode="markers",
            marker=dict(size=marker_size, color=colors_before, opacity=0.8),
            name="Before",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Right – after
    fig.add_trace(
        go.Scatter(
            x=X_tsne_clean[:, 0],
            y=X_tsne_clean[:, 1],
            mode="markers",
            marker=dict(size=marker_size, color="steelblue", opacity=0.8),
            name="After",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Axis labels
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)

    fig.update_layout(
        title_text="t-SNE: Outlier Removal (DBSCAN on t-SNE embedding)",
        height=height,
        width=width,
        template="plotly_white",
    )

    fig.show()

    # ------------------------------------------------------------------ #
    # 5. Print short report
    # ------------------------------------------------------------------ #
    n_outliers = is_outlier.sum()
    print(f"Original samples : {X_full.shape[0]:,}")
    print(f"Outliers removed : {n_outliers:,} "
          f"({n_outliers / X_full.shape[0]:.2%})")
    print(f"Clean samples    : {X_clean.shape[0]:,}")

if __name__ == "__main__":
    # Load data
    df = pl.read_csv("data/merged3.csv", schema_overrides={"isbn": pl.Utf8})
    sales = (pl.read_csv("expand_quarterly_sales_retail.csv",
                         schema_overrides={"barcode2": pl.Utf8, "Quarter_num": pl.Utf8})
               .rename({"barcode2": "isbn"}))

    unwanted_cols = ["isbn", "title", "author", "Next_Q1", "Next_Q2", "Next_Q3", "Next_Q4",
                     "Next_Q2_log1p", "Next_Q3_log1p", "Next_Q4_log1p",
                     "Current_quarter", "First_day", "Avg_discount"]

    # Preprocess
    X_train, X_test, _, _ = preprocess_data(books_df=df, sales_df=sales)

    # --- after preprocessing -------------------------------------------------
    feats = [c for c in X_train.columns
             if c != "Next_Q1_log1p" and c not in unwanted_cols]

    X_np = X_train.select(feats).to_numpy().astype(np.float32)
    print(f"X_np type: {type(X_np)}, shape: {X_np.shape}")  # Debug: should be <class 'numpy.ndarray'>

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=40, init="pca",
                learning_rate="auto", random_state=42, n_jobs=-1)
    X_2d = np.asarray(tsne.fit_transform(X_np))  # Ensure NumPy
    print(f"X_2d type: {type(X_2d)}, shape: {X_2d.shape}")  # Debug: should be <class 'numpy.ndarray'>

    # DBSCAN on the 2-D embedding
    db = DBSCAN(eps=3.0, min_samples=10)  # tune eps after first plot
    labels = db.fit_predict(X_2d)  # -1 = outlier
    inlier_mask = np.asarray(labels != -1)  # Ensure NumPy bool array
    print(f"inlier_mask type: {type(inlier_mask)}, dtype: {inlier_mask.dtype}")  # Debug: bool

    n_outliers = (~inlier_mask).sum()
    print(f"Removed {n_outliers:,} outliers ({n_outliers / len(labels):.2%})")

    # Plot – now works perfectly
    inliers_x = X_2d[inlier_mask][:, 0]
    inliers_y = X_2d[inlier_mask][:, 1]
    outliers_x = X_2d[~inlier_mask][:, 0]
    outliers_y = X_2d[~inlier_mask][:, 1]

    # After you have DBSCAN labels (labels == -1 → outlier)
    outlier_mask = labels == -1  # or directly use `~inlier_mask`

    # Convert Polars → NumPy once (you already have X_np)
    X_full_np = X_np  # full data (already imputed & scaled)
    X_clean_np = X_np[inlier_mask]  # only the rows that survived DBSCAN

    # Call the nice Plotly visualisation
    plot_outlier_comparison(
        X_full=X_full_np,
        outlier_mask=~inlier_mask,  # True = outlier
        X_clean=X_clean_np,
        perplexity=40,
    )

    # Apply mask back to Polars DataFrame (convert NumPy mask to Polars Series)
    X_train_clean = X_train.filter(pl.Series(inlier_mask))  # works
    # or: X_train_clean = X_train.filter(pl.from_numpy(inlier_mask))

    # Continue to tensors
    X_tensor, y_tensor, _ = to_tensors(
        df=X_train_clean,
        target_col="Next_Q1_log1p",
        drop_cols=unwanted_cols
    )

    print(f"Final train → X: {X_tensor.shape}, y: {y_tensor.shape}")