import torch
import torch.nn as nn


class SalesPredictor(nn.Module):
    def __init__(
        self,
        text_embedding_dim: int,
        image_embedding_dim: int,
        hidden_dim: int = 64,        # same intermediate size as your AE (64)
        dropout: float = 0.3
    ):
        super().__init__()

        # Total dimension after simple concatenation
        input_dim = text_embedding_dim + image_embedding_dim

        # Exact same architecture pattern as your AutoEncoder's encoder + decoder
        # but ending with output dim = 1 (regression)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # same as AE: input → 64
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, 1)            # final output: predict sales (Next_Q1_log1p)
        )

        # Optional dropout if you want (you can insert it later)
        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        text_emb: torch.Tensor,    # shape: (batch_size, text_embedding_dim)
        image_emb: torch.Tensor    # shape: (batch_size, image_embedding_dim)
    ) -> torch.Tensor:

        # Concatenate the two pre-computed embeddings
        x = torch.cat([text_emb, image_emb], dim=1)   # → (batch_size, input_dim)

        # Optional dropout on concatenated features
        x = self.dropout(x)

        # Same style as your autoencoder: two linear + LeakyReLU, but final dim=1
        x = self.regressor(x)

        return x   # → (batch_size, 1)