import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import AutoEncoder
from sales_predictor import SalesPredictor
import numpy as np
import polars as pl
from pathlib import Path
from typing import Union


@torch.no_grad()
def get_ae_embeddings(
        X: Union[torch.Tensor, np.ndarray],
        weights_path: Union[str, Path],
        input_dim: int,
        encoding_dim: int = 32,
        batch_size: int = 512,
        device: str = None
) -> np.ndarray:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Encoder weights not found: {weights_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to tensor if needed
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    elif isinstance(X, torch.Tensor):
        X = X.float()
    else:
        raise TypeError("X must be numpy array or torch tensor")

    X = X.to(device)

    model = AutoEncoder(input_dim=input_dim, encoding_dim=encoding_dim)
    state_dict = torch.load(weights_path, map_location=device)
    model.encoder.load_state_dict(state_dict)
    model.encoder.eval()

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    for (batch,) in loader:
        emb = model.encoder(batch)
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings)


@torch.no_grad()
def get_sales_embeddings(
        encoded: Union[torch.Tensor, np.ndarray],
        text: Union[torch.Tensor, np.ndarray],
        image: Union[torch.Tensor, np.ndarray],
        weights_path: Union[str, Path],
        dropout: float = 0.2,
        batch_size: int = 512,
        device: str = None
) -> np.ndarray:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Sales predictor weights not found: {weights_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Properly convert all inputs to torch tensors
    def to_tensor(arr, name):
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        elif isinstance(arr, torch.Tensor):
            return arr.float()
        else:
            raise TypeError(f"{name} must be numpy array or torch tensor")

    encoded = to_tensor(encoded, "encoded").to(device)
    text = to_tensor(text, "text").to(device)
    image = to_tensor(image, "image").to(device)

    # Check shapes
    if encoded.ndim != 2 or text.ndim != 2 or image.ndim != 2:
        raise ValueError("All inputs must be 2D (n_samples, dim)")

    n_samples = encoded.shape[0]
    if text.shape[0] != n_samples or image.shape[0] != n_samples:
        raise ValueError("All inputs must have the same number of samples")

    encoded_dim = encoded.shape[1]
    text_dim = text.shape[1]
    image_dim = image.shape[1]

    model = SalesPredictor(
        encoded_dim=encoded_dim,
        text_dim=text_dim,
        image_dim=image_dim,
        dropout=dropout
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = TensorDataset(encoded, text, image)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    for enc_b, txt_b, img_b in loader:
        pred = model(enc_b, txt_b, img_b)
        predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions).flatten()


if __name__ == "__main__":
    data_folder = Path("data")
    results_folder = Path("sales_results")
    results_folder.mkdir(exist_ok=True)

    # --------------------- Load data ---------------------
    train_enriched = pl.read_csv(data_folder / "train_enriched.csv", schema_overrides={"isbn": pl.Utf8})
    test_enriched = pl.read_csv(data_folder / "test_enriched.csv", schema_overrides={"isbn": pl.Utf8})

    target_col = "Next_Q1_log1p"
    img_cols = [f"img{i}" for i in range(2048)]
    text_cols = [f"text{i}" for i in range(384)]
    feat_cols = [col for col in train_enriched.columns
                if col not in img_cols + text_cols + ["isbn", target_col]]

    # --------------------- Extract features as NumPy ---------------------
    X_train_img = train_enriched.select(img_cols).to_numpy()
    X_train_text = train_enriched.select(text_cols).to_numpy()
    X_train_feats = train_enriched.select(feat_cols).to_numpy()

    X_test_img = test_enriched.select(img_cols).to_numpy()
    X_test_text = test_enriched.select(text_cols).to_numpy()
    X_test_feats = test_enriched.select(feat_cols).to_numpy()

    # --------------------- Autoencoder embeddings ---------------------
    print("Generating encoded features with autoencoder...")
    train_ae = get_ae_embeddings(
        X=X_train_feats,
        weights_path="ae_results/encoder_weights.pth",
        input_dim=X_train_feats.shape[1]
    )
    test_ae = get_ae_embeddings(
        X=X_test_feats,
        weights_path="ae_results/encoder_weights.pth",
        input_dim=X_test_feats.shape[1]
    )
    print(f"AE features → train: {train_ae.shape}, test: {test_ae.shape}")

    # --------------------- Sales predictions ---------------------
    print("Generating sales predictions...")
    best_model_path = sorted(results_folder.glob("sales_predictor_best_*.pth"))[-1]
    print(f"Using best model: {best_model_path.name}")

    test_predictions_log1p = get_sales_embeddings(
        encoded=test_ae,
        text=X_test_text,
        image=X_test_img,
        weights_path=results_folder / "sales_predictor_best_epoch_142_lr_1e-04_bs_32.pth",
        dropout=0.2,
        batch_size=512
    )

    test_predictions = np.expm1(test_predictions_log1p)

    print(f"Generated {len(test_predictions)} predictions")
    print(f"  Mean predicted sales: {test_predictions.mean():.2f}")
    print(f"  Max predicted sales:  {test_predictions.max():.2f}")

    # --------------------- Save submission ---------------------
    submission = pl.DataFrame({
        "isbn": test_enriched["isbn"].to_numpy(),
        "Next_Q1": np.clip(test_predictions, 0, None)
    })

    submission_path = results_folder / "submission.csv"
    # submission.write_csv(submission_path)
    print(f"\nSubmission successfully saved to: {submission_path}")