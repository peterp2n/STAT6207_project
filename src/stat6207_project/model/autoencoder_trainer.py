import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import numpy as np

from autoencoder import AutoEncoder


class AutoEncoderTrainer(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32, lr: float = 0.001):
        super().__init__()
        self.model = AutoEncoder(input_dim, encoding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Store losses for plotting
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train(self,
              train_data: torch.Tensor,
              epochs: int = 50,
              batch_size: int = 64,
              val_data: Optional[torch.Tensor] = None,
              print_every: int = 10):
        """
        Train the autoencoder and record losses.
        """
        dataset = TensorDataset(train_data, train_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, epochs + 1):
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
            self.train_losses.append(avg_train_loss)

            # Validation loss
            val_mse = None
            if val_data is not None:
                val_mse = self._compute_mse(val_data)
                self.val_losses.append(val_mse)
            else:
                self.val_losses.append(None)

            # Logging
            if epoch % print_every == 0 or epoch == epochs:
                log_str = f"Epoch [{epoch:>3}/{epochs}]  Train MSE: {avg_train_loss:.6f}"
                if val_mse is not None:
                    log_str += f"  |  Val MSE: {val_mse:.6f}"
                print(log_str)

    def _compute_mse(self, data: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, data)
        self.model.train()
        return loss.item()

    def evaluate(self, test_data: torch.Tensor) -> float:
        return self._compute_mse(test_data)

    def get_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(data)
        return embeddings

    def plot_losses(self,
                    figsize: Tuple[int, int] = (10, 6),
                    title: str = "Autoencoder Training & Validation MSE",
                    save_path: Optional[str] = None,
                    y_scale: str = 'linear',  # 'linear' or 'log'
                    y_lim: Optional[Tuple[float, float]] = None,
                    y_ticks: Optional[List[float]] = None,
                    y_tick_labels: Optional[List[str]] = None):
        """
        Plot Train and Validation MSE vs Epochs with full y-axis control.

        Args:
            figsize: Figure size
            title: Plot title
            save_path: If provided, saves the plot
            y_scale: 'linear' (default) or 'log' for logarithmic y-axis
            y_lim: (min, max) tuple to set y-axis limits, e.g., (1e-6, 1.0)
            y_ticks: List of specific tick locations, e.g., [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0]
            y_tick_labels: Optional custom labels for y_ticks
        """
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=figsize)

        # Plot training loss
        plt.plot(epochs, self.train_losses, label="Train MSE", color="tab:blue", linewidth=2.5)

        # Plot validation loss if available
        if self.val_losses and any(v is not None for v in self.val_losses):
            val_epochs = [i + 1 for i, v in enumerate(self.val_losses) if v is not None]
            val_losses_clean = [v for v in self.val_losses if v is not None]
            plt.plot(val_epochs, val_losses_clean, label="Validation MSE", color="tab:orange", linewidth=2.5,
                     linestyle="--")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.title(title, fontsize=14, pad=15)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which='both', linestyle='--')

        # === Y-Axis Customization ===
        plt.yscale(y_scale)  # 'linear' or 'log'

        if y_lim is not None:
            plt.ylim(y_lim)

        if y_ticks is not None:
            plt.yticks(y_ticks, y_tick_labels if y_tick_labels else [f"{t:.0e}" for t in y_ticks])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curve saved → {save_path}")

        plt.show()

    def save_weights(self, part: str, path: str):
        """
        Save the weights (state_dict) of either the encoder or decoder to a file.

        Args:
            part: 'encoder' or 'decoder' to specify which part's weights to save.
            path: File path to save the weights (e.g., 'encoder_weights.pth').
        """
        if part == 'encoder':
            torch.save(self.model.encoder.state_dict(), path)
            print(f"Encoder weights saved to {path}")
        elif part == 'decoder':
            torch.save(self.model.decoder.state_dict(), path)
            print(f"Decoder weights saved to {path}")
        else:
            raise ValueError("part must be 'encoder' or 'decoder'")