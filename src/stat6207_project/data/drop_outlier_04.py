from pathlib import Path
import polars as pl
import polars.selectors as cs
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


def plot_outlier_comparison(X_imputed, outlier_labels, X_clean):
    """
    Create side-by-side t-SNE visualization before and after outlier removal.

    Parameters:
    -----------
    X_imputed : np.ndarray
        Full imputed dataset (before outlier removal)
    outlier_labels : np.ndarray
        Outlier predictions from IsolationForest (1=inlier, -1=outlier)
    X_clean : np.ndarray
        Cleaned dataset (after outlier removal)
    """
    # Run t-SNE on ALL data (before removal)
    tsne_full = TSNE(n_components=2, random_state=42)
    X_tsne_full = tsne_full.fit_transform(X_imputed)

    # Run t-SNE on cleaned data (after removal)
    tsne_clean = TSNE(n_components=2, random_state=42)
    X_tsne_clean = tsne_clean.fit_transform(X_clean)

    # Create side-by-side plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Before (n={len(X_imputed)}, outliers highlighted)',
            f'After (n={len(X_clean)}, outliers removed)'
        )
    )

    # Left plot: Before removal (with outliers highlighted)
    colors_before = ['red' if label == -1 else 'blue' for label in outlier_labels]
    fig.add_trace(
        go.Scatter(
            x=X_tsne_full[:, 0],
            y=X_tsne_full[:, 1],
            mode='markers',
            marker=dict(size=5, color=colors_before),
            showlegend=False
        ),
        row=1, col=1
    )

    # Right plot: After removal (only inliers)
    fig.add_trace(
        go.Scatter(
            x=X_tsne_clean[:, 0],
            y=X_tsne_clean[:, 1],
            mode='markers',
            marker=dict(size=5, color='blue'),
            showlegend=False
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="First t-SNE", row=1, col=1)
    fig.update_yaxes(title_text="Second t-SNE", row=1, col=1)
    fig.update_xaxes(title_text="First t-SNE", row=1, col=2)
    fig.update_yaxes(title_text="Second t-SNE", row=1, col=2)

    fig.update_layout(
        title_text="t-SNE Visualization: Before vs After Outlier Removal",
        height=500,
        width=1200
    )

    fig.show()

    # Print statistics
    n_outliers = np.sum(outlier_labels == -1)
    print(f"Original samples: {len(X_imputed)}")
    print(f"Outliers detected: {n_outliers} ({n_outliers / len(X_imputed) * 100:.2f}%)")
    print(f"Samples after removal: {len(X_clean)}")


if __name__ == "__main__":
    data = Path("data")

    # Load data
    merge5_std_dummy = (
        pl.scan_csv(
            data / "merged4_std_dummy.csv", schema_overrides={"isbn": pl.Utf8}
        )
        .collect()
    )

    merge5_std_dummy_numeric = (
        merge5_std_dummy
        .select(cs.numeric())
    )

    # Convert Polars DataFrame to NumPy array for sklearn
    X = merge5_std_dummy_numeric.to_numpy()

    # Apply IterativeImputer
    imputer = IterativeImputer(random_state=42, max_iter=20)
    X_imputed = imputer.fit_transform(X)

    # Detect outliers
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(X_imputed)

    # Filter to keep only inliers
    X_clean = X_imputed[outlier_labels == 1]

    # Plot comparison (single line call)
    plot_outlier_comparison(X_imputed, outlier_labels, X_clean)

    merge5_std_dummy_drop = (
        merge5_std_dummy
        .filter(outlier_labels == 1)
    )
    # merge5_std_dummy_drop.write_csv(data / "merged5_std_dummy.csv", include_bom=True)

    pass