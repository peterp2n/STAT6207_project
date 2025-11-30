import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoEncoder


class AutoEncoderTrainer(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32, lr: float = 0.001):
        super().__init__()
        self.model = AutoEncoder(input_dim, encoding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, train_data: torch.Tensor, epochs: int = 50, batch_size: int = 64, val_data: torch.Tensor = None):
        """
        Train the autoencoder, with optional validation logging per epoch.

        Args:
            train_data: Torch tensor [N, input_dim] (e.g., preprocessed book features/sales).
            epochs: Number of epochs.
            batch_size: Batch size.
            val_data: Optional torch tensor [M, input_dim] for validation MSE logging.
        """
        dataset = TensorDataset(train_data, train_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_train_loss = 0.0
            for batch in loader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(loader)
            log_str = f"Epoch [{epoch + 1}/{epochs}], Train MSE: {avg_train_loss:.4f}"

            if val_data is not None:
                val_mse = self._compute_mse(val_data)  # Internal helper for eval
                log_str += f", Val MSE: {val_mse:.4f}"

            print(log_str)

    def evaluate(self, test_data: torch.Tensor) -> float:
        """Evaluate reconstruction MSE on test data."""
        return self._compute_mse(test_data)

    def _compute_mse(self, data: torch.Tensor) -> float:
        """Internal helper to compute MSE without gradients."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, data)
        self.model.train()  # Reset to train mode if needed
        return loss.item()

    def get_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(data)
        return embeddings