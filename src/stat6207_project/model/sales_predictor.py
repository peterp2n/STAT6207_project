import torch
import torch.nn as nn


class SalesPredictor(nn.Module):
    def __init__(
        self,
        encoded_dim: int = 32,
        text_dim: int = 384,
        image_dim: int = 2048,
        dropout: float = 0.2
    ):
        super().__init__()
        total_dim = encoded_dim + text_dim + image_dim

        self.net = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Linear(64, 1)   # predicts Next_Q1_log1p directly
        )

    def forward(self, encoded_emb: torch.Tensor, text_emb: torch.Tensor, image_emb: torch.Tensor):
        x = torch.cat([encoded_emb, text_emb, image_emb], dim=1)
        return self.net(x).squeeze(-1)  # (batch_size,)