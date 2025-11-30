import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoEncoder  # Assuming this is in a separate file


class AutoEncoderTrainer(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32, lr: float = 0.001):
        """
        Initialize the trainer with the autoencoder model.

        Args:
            input_dim: Dimensionality of the input features (e.g., from book metadata or flattened sales history).
            encoding_dim: Size of the encoded latent space (default 32 for compression).
            lr: Learning rate for the optimizer.
        """
        super().__init__()
        self.model = AutoEncoder(input_dim, encoding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, train_data: torch.Tensor, epochs: int = 50, batch_size: int = 64):
        """
        Train the autoencoder on the provided data (unsupervised reconstruction).

        Args:
            train_data: Torch tensor of shape [N, input_dim] (e.g., from X_train.npy).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        # Create DataLoader for batching (input == target for reconstruction)
        dataset = TensorDataset(train_data, train_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

    def evaluate(self, test_data: torch.Tensor) -> float:
        """
        Evaluate reconstruction loss on test data.

        Args:
            test_data: Torch tensor of shape [M, input_dim] (e.g., from X_test.npy).

        Returns:
            MSE loss on the test data.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_data)
            loss = self.criterion(outputs, test_data)
        return loss.item()

    def get_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get encoded embeddings for input data (useful for feature extraction in sales prediction).

        Args:
            data: Torch tensor of shape [K, input_dim].

        Returns:
            Encoded tensor of shape [K, encoding_dim].
        """
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(data)
        return embeddings