import torch
import torch.nn as nn


class SalesPredictor(nn.Module):
    def __init__(
        self,
        encoded_embedding_dim: int,
        text_embedding_dim: int,
        image_embedding_dim: int,
        dropout: float = 0.3
    ):
        super().__init__()

        # Total dimension after simple concatenation
        input_dim = encoded_embedding_dim + text_embedding_dim + image_embedding_dim

        # Exact same architecture pattern as your AutoEncoder's encoder + decoder
        # but ending with output dim = 1 (regression)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),   # same as AE: input → 64
            nn.LeakyReLU(inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1)            # final output: predict sales (Next_Q1_log1p)
        )

        # Optional dropout if you want (you can insert it later)
        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        encoded_emb: torch.Tensor,  # shape: (batch_size, encoded_embedding_dim)  # unused for now
        text_emb: torch.Tensor,    # shape: (batch_size, text_embedding_dim)
        image_emb: torch.Tensor    # shape: (batch_size, image_embedding_dim)
    ) -> torch.Tensor:

        # Concatenate the two pre-computed embeddings
        x = torch.cat([encoded_emb, text_emb, image_emb], dim=1)   # → (batch_size, input_dim)

        # Same style as your autoencoder: two linear + LeakyReLU, but final dim=1
        x = self.regressor(x)

        return x   # → (batch_size, 1)