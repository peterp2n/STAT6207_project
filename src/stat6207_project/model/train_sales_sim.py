# train_sales_sim.py
# "Less is more. Strip everything that doesn't move the needle." – Elon

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
import polars as pl
import numpy as np

from encode_features import get_embeddings  # reuse your proven encoder loader
from sales_predictor_trainer_sim import SalesPredictorTrainerSim


class BookDatasetSim(Dataset):
    """Only the encoded 32-dim embedding + target. Pure signal."""
    def __init__(self, encoded: torch.Tensor, target: torch.Tensor):
        self.encoded = encoded.float()          # (N, 32)
        self.target = target.float().squeeze()  # (N,)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx], self.target[idx]


if __name__ == "__main__":
    data_folder = Path("data")
    results_folder = Path("sales_results_sim")
    results_folder.mkdir(exist_ok=True)

    print("Loading data...")
    df = pl.read_csv(data_folder / "train_enriched.csv")

    # Extract structured features → compress with trained AE
    img_cols  = [c for c in df.columns if c.startswith("img")]
    text_cols = [c for c in df.columns if c.startswith("text")]
    feat_cols = [c for c in df.columns
                 if c not in img_cols + text_cols + ["isbn", "Next_Q1_log1p"]]

    X_feats = df.select(feat_cols).to_numpy()
    y = torch.from_numpy(df["Next_Q1_log1p"].to_numpy()).float()

    print(f"Raw structured features: {X_feats.shape[1]} → compressing to 32-dim...")

    # One-shot encoding with your trained autoencoder
    X_encoded = get_embeddings(
        X=X_feats,
        weights_path="ae_results/encoder_weights.pth",
        input_dim=X_feats.shape[1],
        encoding_dim=32,
        batch_size=1024
    )  # → torch.Tensor (N, 32) on CPU

    print(f"Encoded shape: {X_encoded.shape}")

    # Dataset & split
    dataset = BookDatasetSim(X_encoded, y)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 128          # larger batches = faster + more stable gradients
    epochs = 1000
    lr = 3e-4                 # slightly higher LR – small model converges fast
    patience = 40
    print_every = 10

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training set: {len(train_ds):,} | Validation set: {len(val_ds):,}")
    print(f"Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")

    trainer = SalesPredictorTrainerSim(
        encoded_dim=32,
        lr=lr,
        weight_decay=1e-5,
        dropout=0.2,
        use_layernorm=True,
        leaky_slope=0.01
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        print_every=print_every
    )

    # Save loss curves
    trainer.plot_losses(
        save_path=results_folder / f"sim_loss_curve_lr_{lr}_bs_{batch_size}.png"
    )

    print("Training complete. Best model saved in sales_results_sim/")
    print("Now go ship it.")