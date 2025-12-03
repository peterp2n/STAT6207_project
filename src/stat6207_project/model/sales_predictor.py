import torch
import torch.nn as nn


class SalesPredictor(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        text_dim: int,
        image_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()

        # Separate learnable branches (unchanged)
        self.encoded_branch = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.ReLU(inplace=True)
        )
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(inplace=True)
        )
        self.image_branch = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.ReLU(inplace=True)
        )

        # Total concatenated dimension
        input_dim = encoded_dim + text_dim + image_dim

        # Even wider regressor with added 1024 layer before 512
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Linear(64, 1)
        )

    def forward(
        self,
        encoded_emb: torch.Tensor,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoded_branch(encoded_emb)
        text    = self.text_branch(text_emb)
        image   = self.image_branch(image_emb)

        x = torch.cat([encoded, text, image], dim=1)   # (B, input_dim)
        x = self.regressor(x)

        return x.squeeze(-1)   # (B,)