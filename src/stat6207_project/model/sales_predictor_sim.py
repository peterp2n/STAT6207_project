import torch
import torch.nn as nn


class SalesPredictor(nn.Module):
    def __init__(
        self,
        encoded_dim: int = 32,
        dropout: float = 0.2,           # reduced from 0.3 — less regularization needed with small model
        use_layernorm: bool = True,     # LayerNorm works better than BatchNorm for small inputs/batches
        leaky_slope: float = 0.01
    ):
        super().__init__()

        # Prefer LayerNorm when input dim is small (32) and batches may vary in size
        Norm = nn.LayerNorm if use_layernorm else nn.BatchNorm1d

        # Optional: tiny refinement of the 32-dim embedding
        self.preprocess = nn.Sequential(
            nn.Linear(encoded_dim, 64),        # small expansion for more capacity
            nn.ReLU(inplace=True),
            Norm(64),
            nn.Dropout(dropout * 0.5)          # light dropout early
        )

        # Shallow but effective regressor: 64 → 64 → 32 → 1
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            Norm(64),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            Norm(32),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1)                   # raw logit output for regression
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming init works well with ReLU/LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, encoded_emb: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(encoded_emb)   # (batch_size, 64)
        x = self.regressor(x)              # (batch_size, 1)
        return x.squeeze(-1)               # (batch_size,)