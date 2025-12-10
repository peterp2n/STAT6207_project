import torch.nn as nn


class RegressorMini(nn.Module):
    def __init__(self, input_dim=10, dropout=0.3):
        super().__init__()

        # Simplified network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)