import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

# -------------------------- Device Setup --------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "mps":
    print("MPS (Apple Silicon GPU) is active!\n")

# -------------------------- Autoencoder Model --------------------------
class Autoencoder(nn.Module):
    def __init__(self, df: np.ndarray, encoding_dim: int = 32):
        super().__init__()
        self.data = torch.from_numpy(df)
        input_dim = df.shape[1]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)



if __name__ == "__main__":
    # Load data
    data_folder = Path("data")
    data_df = pd.read_csv(
        data_folder / "merged5_all.csv",
        dtype={"isbn": "string"}
    ).drop(columns=["reading_age_baby"], errors="ignore")

    numeric_df = data_df.select_dtypes(include="number").to_numpy().astype(np.float32)
    input_dim = numeric_df.shape[1]


    class AutoencoderTrainer:

        def __init__(self, df: np.ndarray, batch_size: int, split: float):

            self.train_size = int(split * len(df))
            self.test_size = len(df) - self.train_size
            self.batch_size = batch_size

            self.train_dataset, self.test_dataset = random_split(
                numeric_df,
                [self.train_size, self.test_size],
                generator=torch.Generator().manual_seed(42)  # reproducible split
            )

            num_workers = 0

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

        def train(self,
                  model,
                  optimizer,
                  batch_size,
                  num_epochs,
                  learning_rate,
                  encoding_dim,
                  weight
            ):





        # Model, loss, optimizer
        model = Autoencoder(df=numeric_df, encoding_dim=32).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        num_epochs = 100
        model.train()




    pass