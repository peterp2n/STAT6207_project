import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, input_dim=10, dropout=0.3):
        super().__init__()

        # Simplified network architecture
        self.net = nn.Sequential(
            # Reduced the first layer to 32 neurons from 64
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Removed the second hidden layer entirely
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)