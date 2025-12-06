# autoencoder.py
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()

        # Auto-detect device: MPS > CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"AutoEncoder initialized on {self.device}")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, encoding_dim),
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, input_dim),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)  # Ensures input is on correct device
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.encoder(x)