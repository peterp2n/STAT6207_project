import polars as pl
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Optional, Union
import warnings
from scipy.spatial.distance import pdist  # Added import

from autoencoder import AutoEncoder  # Assuming this is your module


def visualize_isbn_group(
        df: pl.DataFrame,
        isbn_list: Optional[Union[str, List[str]]] = None,
        weights_path: str = "ae_results/encoder_weights.pth",
        num_background: int = 2000,
        method: str = "tsne",
        perplexity: int = 30,
        random_state: int = 42,
        title_prefix: str = "Autoencoder Latent Space",
        marker_size: int = 90,
        connect_same_isbn: bool = True,
        compute_distances: bool = True,
        save_path: Optional[str] = None,
):
    if "isbn" not in df.columns:
        raise ValueError("DataFrame must contain 'isbn' column")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === DYNAMICALLY INFER INPUT DIM FROM SAVED ENCODER WEIGHTS ===
    try:
        # Load only the state_dict to inspect architecture without knowing input_dim
        state_dict = torch.load(weights_path, map_location=device)

        # The first linear layer in encoder is Linear(input_dim → 64)
        first_layer_weight = state_dict[list(state_dict.keys())[0]]  # e.g. "0.weight"
        input_dim = first_layer_weight.shape[1]  # This is our true input dimension!

        print(f"Detected encoder input dimension: {input_dim} features")
    except Exception as e:
        raise FileNotFoundError(f"Could not load or inspect encoder weights at {weights_path}: {e}")

    # Now build the model with the correct (dynamic) input_dim
    model = AutoEncoder(input_dim=input_dim, encoding_dim=32)
    try:
        model.encoder.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Encoder architecture mismatch when loading weights: {e}")

    model.eval()
    model.to(device)

    # Extract features (exclude 'isbn')
    feature_cols = [col for col in df.columns if col != "isbn"]
    actual_num_features = len(feature_cols)

    if actual_num_features != input_dim:
        raise ValueError(
            f"Feature count mismatch!\n"
            f"   Model expects {input_dim} features, but DataFrame has {actual_num_features}.\n"
            f"   Check your preprocessing pipeline or retrain the autoencoder."
        )
    else:
        print(f"Perfect match: Data has {actual_num_features} features → matches model input dim")

    features = df.select(feature_cols).to_numpy().astype(np.float32)  # (N, input_dim)

    # Handle ISBN list
    if isbn_list is not None:
        if isinstance(isbn_list, str):
            isbn_list = [isbn_list]
        isbn_list = [str(isbn) for isbn in isbn_list]  # Cast once
        mask = df["isbn"].is_in(isbn_list)
        group_df = df.filter(mask)
        group_indices = mask.arg_true().to_numpy()
        group_features = features[group_indices]
        group_isbns = group_df["isbn"].to_list()

        found_count = len(group_features)
        print(f"Found {found_count} rows for ISBN(s): {set(isbn_list)}")
        if found_count == 0:
            warnings.warn("No rows found for provided ISBN(s). Plotting only background.")
    else:
        group_features = None
        group_isbns = []
        group_indices = np.array([])  # Empty for exclusion

    # Sample background (exclude highlighted)
    n_total = len(df)
    exclude_indices = set(group_indices)
    available_indices = list(set(range(n_total)) - exclude_indices)
    if len(available_indices) < num_background:
        bg_indices = available_indices
        print(f"Using all {len(bg_indices)} available background points")
    else:
        bg_indices = np.random.choice(available_indices, num_background, replace=False)

    bg_features = features[bg_indices]

    # Encode
    def encode_batch(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return np.empty((0, 32))  # Empty array handling
        with torch.no_grad():
            tensor = torch.from_numpy(arr).to(device)
            return model.encode(tensor).cpu().numpy()

    bg_emb = encode_batch(bg_features)
    group_emb = encode_batch(group_features) if group_features is not None else np.empty((0, 32))

    # Combine for reduction
    all_emb = np.vstack([bg_emb, group_emb]) if len(group_emb) > 0 else bg_emb
    if len(all_emb) < 2:
        warnings.warn("Fewer than 2 points total; skipping dimensionality reduction and plot.")
        return

    # 2D projection
    if method == "tsne":
        perp = min(perplexity, max(5, len(all_emb) // 4))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=random_state, init="pca", learning_rate="auto")
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    all_2d = reducer.fit_transform(all_emb)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    bg_2d = all_2d[:len(bg_emb)]
    ax.scatter(bg_2d[:, 0], bg_2d[:, 1], c="lightgray", alpha=0.5, s=30, label="Background")

    if len(group_emb) > 0:
        group_2d = all_2d[len(bg_emb):]
        unique_isbns = list(set(isbn_list))  # In case duplicates in input
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_isbns)))
        color_map = dict(zip(unique_isbns, colors))

        for isbn in unique_isbns:
            idx = [j for j, g_isbn in enumerate(group_isbns) if g_isbn == isbn]
            if not idx:
                continue  # Skip if no points (warned earlier)
            pts = group_2d[idx]
            color = color_map[isbn]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=[color], s=marker_size, label=f"{isbn} ({len(idx)} quarters)",
                edgecolors="black", linewidth=1.2, zorder=5
            )
            if connect_same_isbn and len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.7, linewidth=2.5, zorder=4)

    ax.set_title(f"{title_prefix}\n32D → 2D via {method.upper()}", fontsize=16, pad=20)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

    # Print distances (optional)
    if compute_distances and len(group_emb) > 0:
        print("\n" + "=" * 70)
        print("EUCLIDEAN DISTANCES IN 32D LATENT SPACE (lower = more consistent representation)")
        print("=" * 70)
        for isbn in set(isbn_list):
            idx = [j for j, g_isbn in enumerate(group_isbns) if g_isbn == isbn]
            if len(idx) > 1:
                embs = group_emb[idx]
                dists = pdist(embs, metric="euclidean")
                print(f"{isbn}: {len(idx):2d} quarters → "
                      f"mean={dists.mean():.4f}, std={dists.std():.4f}, "
                      f"min={dists.min():.4f}, max={dists.max():.4f}")

if __name__ == "__main__":
    print("Loading data with ISBNs...")
    df_train = pl.read_csv("data/X_train_with_isbn.csv", schema_overrides={"isbn": pl.Utf8})

    # ————————————————————————————————————————————————————————————————
    #  FOUR VERY SIMILAR BOARD BOOKS – PERFECT COMPARISON
    # ————————————————————————————————————————————————————————————————
    # All are Phidal board books, toddler age, ~22–24 pages, similar price/discount behavior
    four_similar_board_books = [
        "9782764351444",   # Very high & stable sales
        "9782764334591",   # Similar but slightly declining
        "9782764349281",   # Strong seasonal spikes
        "9782764349298",   # Another close cousin
    ]

    print(f"\nComparing 4 extremely similar board books:")
    for isbn in four_similar_board_books:
        n = len(df_train.filter(pl.col("isbn") == isbn))
        print(f"  • {isbn}  →  {n} quarters in training data")

    visualize_isbn_group(
        df=df_train,
        isbn_list=four_similar_board_books,
        weights_path="ae_results/encoder_weights.pth",
        num_background=3000,
        method="tsne",                    # Try "pca" too — both reveal different things!
        perplexity=40,
        marker_size=130,
        connect_same_isbn=True,
        compute_distances=True,
        title_prefix="Latent Space: 4 Nearly Identical Board Books",
        save_path="ae_results/4_similar_board_books_comparison_tsne.png"
    )

    # Bonus: same comparison with PCA (often shows cleaner separation on linear trends)
    visualize_isbn_group(
        df=df_train,
        isbn_list=four_similar_board_books,
        method="pca",
        title_prefix="Latent Space: 4 Nearly Identical Board Books (PCA view)",
        marker_size=130,
        connect_same_isbn=True,
        save_path="ae_results/4_similar_board_books_comparison_pca.png"
    )

    print("\nDone! Check the ae_results/ folder for the two comparison plots.")