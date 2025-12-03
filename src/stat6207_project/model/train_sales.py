# train_sales.py
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler

from encode_features import get_ae_embeddings
from sales_predictor_trainer import SalesPredictorTrainer

# Simple dataset (same as before, but minimal)
from torch.utils.data import Dataset

class BookDataset(Dataset):
    def __init__(self, encoded, text, img, target=None):
        self.encoded = encoded
        self.text = text
        self.img = img
        self.target = target.float() if target is not None else None

    def __len__(self): return len(self.encoded)
    def __getitem__(self, i):
        items = (self.encoded[i], self.text[i], self.img[i])
        return (*items, self.target[i]) if self.target is not None else (*items, torch.tensor(0.0))

if __name__ == "__main__":
    data_folder = Path("data")
    Path("sales_results").mkdir(exist_ok=True)

    df = pl.read_csv(data_folder / "train_enriched.csv")

    img_cols  = [c for c in df.columns if c.startswith("img")]
    text_cols = [c for c in df.columns if c.startswith("text")]
    feat_cols = [c for c in df.columns if c not in img_cols + text_cols + ["isbn", "Next_Q1_log1p"]]

    # Extract
    X_img   = torch.from_numpy(df.select(img_cols).to_numpy()).float()
    X_text  = torch.from_numpy(df.select(text_cols).to_numpy()).float()
    X_feats = torch.from_numpy(df.select(feat_cols).to_numpy()).float()
    y       = torch.from_numpy(df["Next_Q1_log1p"].to_numpy()).float()

    # # Scale structured features (must match AE training!)
    # scaler = StandardScaler()
    # X_feats_scaled = torch.from_numpy(scaler.fit_transform(X_feats.numpy())).float()

    # Encode with trained autoencoder
    X_encoded = get_ae_embeddings(
        X=X_feats.numpy(),
        weights_path="ae_results/encoder_weights.pth", # Load the encoder weights
        input_dim=X_feats.shape[1]
    )

    # Dataset + split
    dataset = BookDataset(X_encoded, X_text, X_img, y)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    batch_size = 32
    epochs = 1000
    learning_rate = 1e-4
    dropout = 0.2
    patience = 30
    print_every = 10

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Trainer
    trainer = SalesPredictorTrainer(
        encoded_dim=X_encoded.shape[1],
        text_dim=len(text_cols),
        image_dim=len(img_cols),
        lr=learning_rate,
        dropout=dropout
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        print_every=print_every
    )

    trainer.plot_losses(save_path=Path(f"sales_results/loss_curve_epoch_{epochs}_lr_{learning_rate}_bs_{batch_size}.png"))