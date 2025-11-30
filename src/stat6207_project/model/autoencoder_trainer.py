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

    def train(self,
              train_data: torch.Tensor,
              epochs: int = 50,
              batch_size: int = 64,
              val_data: torch.Tensor = None,
              print_every: int = 10):  # <-- new argument
        """
        Train the autoencoder.
        Logs train (and validation) MSE only every `print_every` epochs.
        """
        dataset = TensorDataset(train_data, train_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, epochs + 1):  # start counting at 1
            total_train_loss = 0.0
            for batch in loader:
                inputs, targets = batch  # targets == inputs (reconstruction)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(loader)

            # ------------------------------------------------------------------
            # Print only on the epochs you care about
            # ------------------------------------------------------------------
            if epoch % print_every == 0 or epoch == epochs:  # always print final epoch
                log_str = f"Epoch [{epoch:>3}/{epochs}]  Train MSE: {avg_train_loss:.6f}"

                if val_data is not None:
                    val_mse = self._compute_mse(val_data)
                    log_str += f"  |  Val MSE: {val_mse:.6f}"

                print(log_str)

    def evaluate(self, test_data: torch.Tensor) -> float:
        """Evaluate reconstruction MSE on test data."""
        return self._compute_mse(test_data)

    def _compute_mse(self, data: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, data)
        self.model.train()
        return loss.item()

    def get_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(data)
        return embeddings