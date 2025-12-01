import polars as pl
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Optional, Union, Dict
import warnings
from scipy.spatial.distance import pdist

from autoencoder import AutoEncoder

def get_device() -> torch.device:
    """
    Returns the best available device:
      - Apple Silicon → mps  (Metal Performance Shaders – very fast for PyTorch)
      - NVIDIA GPU    → cuda
      - otherwise     → cpu
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # macOS 12.3+ with PyTorch ≥ 1.12 (or nightly ≥ 2.0)
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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

    device = get_device()

    # ------------------------------------------------------------
    # 1. Load the checkpoint and infer BOTH input_dim and encoding_dim
    # ------------------------------------------------------------
    try:
        state_dict = torch.load(weights_path, map_location=device)

        # The encoder is a Sequential: Linear(in→64), ReLU, Linear(64→encoding_dim)
        # The very last layer in the state_dict is the bottleneck layer
        last_weight_key = None
        for k in state_dict.keys():
            if k.endswith(".weight") and state_dict[k].ndim == 2:
                last_weight_key = k                     # e.g. "2.weight"

        if last_weight_key is None:
            raise RuntimeError("Could not find any Linear layer weights in the checkpoint.")

        # Shape of the last Linear layer: [encoding_dim, 64]
        encoding_dim = state_dict[last_weight_key].shape[0]

        # First layer is Linear(input_dim → 64)
        first_weight = state_dict[list(state_dict.keys())[0]]   # "0.weight"
        input_dim = first_weight.shape[1]

        print(f"Detected encoder architecture from checkpoint:")
        print(f"   → input_dim    = {input_dim}")
        print(f"   → encoding_dim = {encoding_dim}")

    except Exception as e:
        raise FileNotFoundError(f"Problem reading {weights_path}: {e}")

    # ------------------------------------------------------------
    # 2. Build the model with the CORRECT dimensions
    # ------------------------------------------------------------
    model = AutoEncoder(input_dim=input_dim, encoding_dim=encoding_dim)
    model.encoder.load_state_dict(state_dict)   # now it will match perfectly
    model.eval()
    model.to(device)

    # ------------------------------------------------------------
    # 3. Validate that the DataFrame has exactly the expected number of features
    # ------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != "isbn"]
    if len(feature_cols) != input_dim:
        raise ValueError(
            f"Feature count mismatch!\n"
            f"   Model expects {input_dim} features, but DataFrame has {len(feature_cols)}.\n"
            f"   Columns found (excluding isbn): {feature_cols}"
        )

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

def get_normalized_embeddings_with_cosine(
    df: pl.DataFrame,
    isbn_list: Union[str, List[str]],
    weights_path: str = "ae_results/encoder_weights.pth",
) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns for each ISBN:
        - embeddings: torch.Tensor of shape (num_quarters, encoding_dim), L2-normalized rows
        - cosine_sim: torch.Tensor of shape (num_quarters, num_quarters), cosine similarity between all quarters

    Because rows are unit-normalized → cosine similarity = dot product → X @ X.T
    """
    if "isbn" not in df.columns:
        raise ValueError("DataFrame must contain 'isbn' column")

    if isinstance(isbn_list, str):
        isbn_list = [isbn_list]
    isbn_list = [str(isbn) for isbn in isbn_list]

    filtered = df.filter(pl.col("isbn").is_in(isbn_list))
    if filtered.is_empty():
        warnings.warn("No matching quarters found for the given ISBNs.")
        return {}

    device = get_device()
    print(f"Running on: {device}")

    # === Auto-detect input_dim and encoding_dim from saved weights ===
    state_dict = torch.load(weights_path, map_location=device)
    input_dim = state_dict["0.weight"].shape[1]  # first layer: [64, input_dim]
    # find last Linear layer going into bottleneck
    encoding_dim = next(v.shape[0] for k, v in state_dict.items() if k.endswith(".weight") and v.shape[1] == 64)
    print(f"Detected: input_dim={input_dim}, encoding_dim={encoding_dim}")

    # Build and load encoder
    model = AutoEncoder(input_dim=input_dim, encoding_dim=encoding_dim)
    model.encoder.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    feature_cols = [c for c in df.columns if c != "isbn"]
    if len(feature_cols) != input_dim:
        raise ValueError(f"Feature mismatch: model expects {input_dim}, got {len(feature_cols)}")

    result = {}

    for isbn in isbn_list:
        group = filtered.filter(pl.col("isbn") == isbn)
        if group.is_empty():
            warnings.warn(f"ISBN {isbn} has no data — skipping")
            continue

        X = torch.tensor(group.select(feature_cols).to_numpy(), dtype=torch.float32, device=device)

        with torch.no_grad():
            Z = model.encode(X)                              # (T, D)
            Z = torch.nn.functional.normalize(Z, p=2, dim=1)  # L2 normalize → unit rows

        # Cosine similarity matrix via matmul (super fast on GPU/MPS)
        cosine_sim_matrix = Z @ Z.T  # (T, T) — each entry is cosine similarity

        result[isbn] = (Z.cpu(), cosine_sim_matrix.cpu())  # move to CPU only at the end

    return result

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

    print("\nFetching normalized embeddings for sample board books:")
    data = get_normalized_embeddings_with_cosine(
        df_train,
        ["9782764351444", "9782764349281", "9782764349298"]
    )

    for isbn, (emb_matrix, sim_matrix) in data.items():
        T = emb_matrix.shape[0]
        print(f"\nISBN: {isbn}")
        print(f"  → {T} quarters → embedding matrix: {emb_matrix.shape}")
        print(f"  → cosine sim matrix:      {sim_matrix.shape}")
        print(f"  → diagonal (should be ~1.0): {sim_matrix.diag().min():.4f} – {sim_matrix.diag().max():.4f}")
        print(f"  → sample similarities (rounded to 4 decimals):\n{torch.round(sim_matrix[:5, :5], decimals=4)}")

    pass