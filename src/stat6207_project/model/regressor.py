import torch
import torch.nn as nn


class Regressor(nn.Module):
    """
    A simple PyTorch regressor model for predicting book sales quantities.

    Based on the dataset dimensions (45484 samples, 13 columns including target),
    we've identified approximately 10 numerical features (format_hardcover, 4 channels,
    discount_rate, q_num_2/3/4, q_since_first) plus categorical features (isbn with 119 unique values,
    title with 116). For an initial guess, assuming an embedding dimension of ~16 for ISBN (sqrt(119) ≈ 11,
    rounded up for flexibility), the effective input dimension is around 26 (16 embed + 10 numerical).

    Thus, the first hidden layer starts with 64 neurons (a power of 2, roughly 2-3x input dim for good capacity
    without overfitting on 45k samples). Subsequent layers taper down for a funnel-like architecture,
    which is a common heuristic for regression tasks to extract hierarchical features.

    This design prioritizes simplicity and efficiency, allowing easy experimentation with hyperparameters.
    """

    def __init__(self, input_dim: int = 26, hidden_dim: int = 64):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Initial guess: 64 neurons
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Output: single regression value (quantity)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Light dropout for regularization on moderate dataset size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()  # Flatten to scalar for regression