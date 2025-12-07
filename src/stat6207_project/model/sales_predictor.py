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

        # Encoded branch: Keep minimal since it's already compressed (32 dims)
        self.encoded_branch = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.BatchNorm1d(encoded_dim),  # Optional: Stabilizes activations
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2)  # Light dropout to prevent overfitting
        )

        # Text branch: Add layers to reduce from 384 → 256 → 128
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),  # 384 → 192 (or use 256 for gradual reduction)
            nn.BatchNorm1d(text_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(text_dim // 2, text_dim // 3),  # 192 → 128 (adjust as needed)
            nn.BatchNorm1d(text_dim // 3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Image branch: Add layers to reduce from 2048 → 1024 → 512
        self.image_branch = nn.Sequential(
            nn.Linear(image_dim, image_dim // 2),  # 2048 → 1024
            nn.BatchNorm1d(image_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(image_dim // 2, image_dim // 4),  # 1024 → 512
            nn.BatchNorm1d(image_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Updated concatenated dimension
        concat_dim = encoded_dim + (text_dim // 3) + (image_dim // 4)  # e.g., 32 + 128 + 512 = 672

        # Regressor: Adjust input_dim; make it wider/deeper if needed for the smaller input
        self.regressor = nn.Sequential(
            nn.Linear(concat_dim, 512),
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
        text = self.text_branch(text_emb)
        image = self.image_branch(image_emb)

        x = torch.cat([encoded, text, image], dim=1)  # Now smaller dim (e.g., 672)
        x = self.regressor(x)

        return x.squeeze(-1)  # (B,)