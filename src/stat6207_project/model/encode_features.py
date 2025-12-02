import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import AutoEncoder  # assumes autoencoder.py is in the same folder or PYTHONPATH
import numpy as np
import polars as pl
from pathlib import Path
from typing import Union


@torch.no_grad()
def get_embeddings(
        X: Union[torch.Tensor, np.ndarray],
        weights_path: Union[str, Path],
        input_dim: int,
        encoding_dim: int = 32,
        batch_size: int = 512,
        device: str = None
) -> np.ndarray:
    """
    Load pretrained autoencoder encoder and return 32-dim embeddings for any feature matrix.

    Args:
        X: Input data of shape (n_samples, input_dim) - numpy array or torch tensor
        weights_path: Path to your saved encoder_weights.pth
        input_dim: Number of input features (must match training)
        encoding_dim: 32 (default, change only if you trained with different size)
        batch_size: Larger = faster + more VRAM
        device: "cuda" or "cpu" (auto-detected if None)

    Returns:
        embeddings: numpy array of shape (n_samples, encoding_dim)
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Encoder weights not found: {weights_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert input to tensor
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    X = X.to(device)

    # Rebuild the full autoencoder just to load the state dict
    model = AutoEncoder(input_dim=input_dim, encoding_dim=encoding_dim)
    state_dict = torch.load(weights_path, map_location=device)

    # Important: your current save_weights() saves ONLY encoder.state_dict()
    # So we load it directly into the encoder part
    model.encoder.load_state_dict(state_dict)
    encoder = model.encoder
    encoder.eval()

    # Batch inference
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    for (batch,) in loader:
        emb = encoder(batch)
        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0).numpy()

if __name__ == "__main__":
    data_folder = Path("data")

    X_train_enriched = pl.read_csv(data_folder / "X_train_enriched.csv", schema_overrides={"isbn": pl.Utf8})
    X_test_enriched = pl.read_csv(data_folder / "X_test_enriched.csv", schema_overrides={"isbn": pl.Utf8})

    img_cols = [f"img{i}" for i in range(2048)]
    text_cols = [f"text{i}" for i in range(384)]
    feat_cols = [col for col in X_train_enriched.columns if col not in img_cols + text_cols + ["isbn"]]

    X_train_img = X_train_enriched.select(img_cols)
    X_train_text = X_train_enriched.select(text_cols)
    X_train_feats = X_train_enriched.select(feat_cols)

    X_train_img_tensor = torch.from_numpy(X_train_img.to_numpy()).float()
    X_train_text_tensor = torch.from_numpy(X_train_text.to_numpy()).float()
    X_train_feats_tensor = torch.from_numpy(X_train_feats.to_numpy()).float()

    # One-liner → get compressed 32-dim features instantly
    train_ae_features = get_embeddings(
        X=X_train_feats.to_numpy(),
        weights_path="ae_results/encoder_weights.pth",
        input_dim=X_train_feats.to_numpy().shape[1]
    )


    print(train_ae_features.shape)  # → e.g. (45234, 32)