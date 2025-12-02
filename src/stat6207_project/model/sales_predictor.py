import torch
import torch.nn as nn


class SalesPredictor(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        text_dim: int,
        image_dim: int,
        dropout: float = 0.3
    ):
        super().__init__()

        # Define separate branches for each embedding type (Linear + ReLU) to allow learnable weights
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

        # Total dimension after concatenation (dims unchanged)
        input_dim = encoded_dim + text_dim + image_dim

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1)
        )

    def forward(
        self,
        encoded_emb: torch.Tensor,  # shape: (batch_size, encoded_dim)
        text_emb: torch.Tensor,     # shape: (batch_size, text_dim)
        image_emb: torch.Tensor     # shape: (batch_size, image_dim)
    ) -> torch.Tensor:
        # Process each embedding through its branch
        encoded = self.encoded_branch(encoded_emb)
        text = self.text_branch(text_emb)
        image = self.image_branch(image_emb)

        # Concatenate the processed embeddings
        x = torch.cat([encoded, text, image], dim=1)  # → (batch_size, input_dim)

        # Pass through regressor
        x = self.regressor(x)

        return x.squeeze()  # → (batch_size)