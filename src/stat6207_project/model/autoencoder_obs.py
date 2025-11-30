import pandas as pd
import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.stat6207_project.data.impute_utils_03 import impute_book_cover, impute_reading_age, standardize_columns
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

# -------------------------- Device Setup --------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")
if device.type == "mps":
    print("MPS (Apple Silicon GPU) is active!\n")


# -------------------------- Custom Dataset --------------------------
class NumericDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# -------------------------- Autoencoder Model --------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),
            # No activation on final layer for reconstruction (MSE loss expects raw values)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# -------------------------- Trainer Class --------------------------
class AutoencoderTrainer:
    def __init__(
            self,
            data: np.ndarray,
            batch_size: int = 32,
            train_ratio: float = 0.8,
            seed: int = 42
    ):
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        # Create dataset
        dataset = NumericDataset(data)

        # Train / test split
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        # DataLoaders
        num_workers = 0 if device.type == "mps" else 4  # MPS doesn't like num_workers > 0
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )

    def train(
            self,
            model: Autoencoder,
            num_epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5
    ) -> Tuple[List[float], List[float]]:

        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_losses = []
        val_losses = []

        print("Starting training...\n")
        for epoch in range(1, num_epochs + 1):
            # ---------- Training ----------
            model.train()
            epoch_train_loss = 0.0
            for batch in self.train_loader:
                batch = batch.to(device)

                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * batch.size(0)

            avg_train_loss = epoch_train_loss / len(self.train_loader.dataset)
            train_losses.append(avg_train_loss)

            # ---------- Validation ----------
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in self.test_loader:
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    epoch_val_loss += loss.item() * batch.size(0)

            avg_val_loss = epoch_val_loss / len(self.test_loader.dataset)
            val_losses.append(avg_val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        print("Training completed!\n")
        return train_losses, val_losses

    def plot_losses(self, train_losses: List[float], val_losses: List[float]):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, label="Training Loss", linewidth=2)
        plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Autoencoder Training & Validation Loss")

        # Fixed y-axis from 0 to 1000
        plt.ylim(0, 50)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    # ------------------ Load and preprocess data ------------------
    data_folder = Path("data")
    csv_path = data_folder / "merged2_dummy_sales.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    data_df = pl.read_csv(csv_path, schema_overrides={"isbn": pl.Utf8})
    features_df = data_df.select(cs.numeric())
    X = features_df.select([col for col in features_df.columns if col != "Next_Q1"]).to_numpy()
    y = data_df["Next_Q1"].to_numpy()



    # Keep only numeric columns and convert to float32
    numeric_df = data_df.select_dtypes(include="number")
    print(f"Original shape: {data_df.shape} → Numeric features shape: {numeric_df.shape}")

    # Simple imputation of missing values (you can improve this)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    data_array = numeric_df.to_numpy().astype(np.float32)

    input_dim = data_array.shape[1]
    print(f"Input dimension: {input_dim}\n")

    # ------------------ Initialize model and trainer ------------------
    encoding_dim = 32

    autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

    trainer = AutoencoderTrainer(
        data=data_array,
        batch_size=64,  # slightly larger batch often works better
        train_ratio=0.85,
        seed=42
    )

    # ------------------ Train the model ------------------
    train_losses, val_losses = trainer.train(
        model=autoencoder,
        num_epochs=300,
        learning_rate=1e-5,
        weight_decay=1e-5
    )

    # ------------------ Plot results ------------------
    trainer.plot_losses(train_losses, val_losses)

    # ------------------ Save the trained model (optional) ------------------
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    torch.save(autoencoder.state_dict(), model_path / "autoencoder.pth")
    print(f"Model saved to {model_path / 'autoencoder.pth'}")

    # ------------------ Example: encode the entire dataset ------------------
    autoencoder.eval()
    with torch.no_grad():
        data_tensor = torch.from_numpy(data_array).to(device)
        embeddings = autoencoder.encode(data_tensor).cpu().numpy()
        print(f"Encoded representations shape: {embeddings.shape}")

    pass