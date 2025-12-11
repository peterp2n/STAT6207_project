import torch.nn as nn


class RegressorMini(nn.Module):
    def __init__(self, input_dim=10, dropout=0.3):
        super().__init__()

        # Simplified network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)