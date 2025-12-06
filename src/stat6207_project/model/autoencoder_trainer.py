# autoencoder_trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
import numpy as np

from autoencoder import AutoEncoder


class AutoEncoderTrainer:
    def __init__(self, input_dim: int, encoding_dim: int = 32, lr: float = 0.001):
        # Auto-detect best device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"AutoEncoderTrainer using device: {self.device}")

        # Model automatically detects and moves to correct device
        self.model = AutoEncoder(input_dim, encoding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.train_losses: List[float] = []   # Stores MSE
        self.val_losses: List[float] = []     # Stores MSE (or None)

    def _mse_to_rmse(self, mse: float) -> float:
        return np.sqrt(mse)

    def train(self,
              train_data: torch.Tensor,
              epochs: int = 50,
              batch_size: int = 64,
              val_data: Optional[torch.Tensor] = None,
              print_every: int = 10):

        train_data = train_data.to(self.device)
        if val_data is not None:
            val_data = val_data.to(self.device)

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

            avg_train_mse = total_train_loss / len(loader)
            avg_train_rmse = self._mse_to_rmse(avg_train_mse)
            self.train_losses.append(avg_train_mse)

            val_mse = None
            val_rmse = None
            if val_data is not None:
                val_mse = self._compute_mse(val_data)
                val_rmse = self._mse_to_rmse(val_mse)
                self.val_losses.append(val_mse)
            else:
                self.val_losses.append(None)

            if epoch % print_every == 0 or epoch == epochs:
                log_str = (f"Epoch [{epoch:>3}/{epochs}]  "
                           f"Train MSE: {avg_train_mse:.6f}  |  Train RMSE: {avg_train_rmse:.6f}")
                if val_mse is not None:
                    log_str += f"  |  Val MSE: {val_mse:.6f}  |  Val RMSE: {val_rmse:.6f}"
                print(log_str)

    def _compute_mse(self, data: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, data)
        self.model.train()
        return loss.item()

    def evaluate(self, test_data: torch.Tensor) -> Tuple[float, float]:
        """
        Returns (MSE, RMSE) on the test set.
        """
        test_data = test_data.to(self.device)
        mse = self._compute_mse(test_data)
        rmse = self._mse_to_rmse(mse)
        return mse, rmse

    def get_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(data)
        self.model.train()
        return embeddings

    def plot_losses(self,
                    figsize: Tuple[int, int] = (10, 6),
                    title: str = "Autoencoder Training & Validation Loss",
                    save_path: Optional[str] = None,
                    y_scale: str = 'auto',
                    y_lim: Union[str, Tuple[float, float], None] = 'auto',
                    y_ticks: Optional[List[float]] = None,
                    y_tick_labels: Optional[List[str]] = None,
                    show_rmse: bool = False):  # New option

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=figsize)

        # Decide whether to plot MSE or RMSE
        train_values = [self._mse_to_rmse(v) if show_rmse else v for v in self.train_losses]
        ylabel = "RMSE Loss" if show_rmse else "MSE Loss"
        plot_title = title.replace("Loss", ylabel.split()[0] + " Loss")

        plt.plot(epochs, train_values, label=f"Train {ylabel}", color="tab:blue", linewidth=2.5)

        if self.val_losses and any(v is not None for v in self.val_losses):
            val_values = [self._mse_to_rmse(v) if show_rmse and v is not None else v
                          for v in self.val_losses]
            val_epochs = [i + 1 for i, v in enumerate(self.val_losses) if v is not None]
            val_clean = [v for v in val_values if v is not None]
            plt.plot(val_epochs, val_clean, label=f"Validation {ylabel}",
                     color="tab:orange", linewidth=2.5, linestyle="--")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(plot_title, fontsize=14, pad=15)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which='both', linestyle='--')

        # Smart dynamic scaling (same logic, now respects RMSE if requested)
        if y_scale == 'auto' or y_lim == 'auto':
            arr = np.array(train_values)
            min_val = arr.min()
            max_val = arr.max()

            if max_val / max(min_val, 1e-8) > 100:
                final_y_scale = 'log'
                final_y_lim = (max(min_val * 0.5, 1e-8), max_val * 2)
            else:
                final_y_scale = 'linear'
                final_y_lim = (0, max_val * 1.1)

            print(f"Auto-detected y_scale = '{final_y_scale}' | y_lim = {final_y_lim}")
        else:
            final_y_scale = y_scale
            final_y_lim = y_lim

        plt.yscale(final_y_scale)
        if final_y_lim is not None and final_y_lim != 'auto':
            plt.ylim(final_y_lim)

        if y_ticks is not None:
            plt.yticks(y_ticks, y_tick_labels)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curve saved → {save_path}")
        plt.show()

    def save_weights(self, part: str, path: str):
        if part == 'encoder':
            torch.save(self.model.encoder.state_dict(), path)
            print(f"Encoder weights saved to {path}")
        elif part == 'decoder':
            torch.save(self.model.decoder.state_dict(), path)
            print(f"Decoder weights saved to {path}")
        else:
            raise ValueError("part must be 'encoder' or 'decoder'")

    def get_reconstruction_similarity(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data)
        orig_norm = torch.nn.functional.normalize(data, p=2, dim=1)
        recon_norm = torch.nn.functional.normalize(reconstructed, p=2, dim=1)
        cos_sim = torch.sum(orig_norm * recon_norm, dim=1)
        return cos_sim


def load_encoder_weights(input_dim: int,
                         encoding_dim: int = 8,
                         weights_path: str = "results/encoder_weights.pth",
                         device: str = "cpu") -> torch.nn.Module:
    """
    Loads a standalone encoder from saved weights.
    """
    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 16),
        torch.nn.LeakyReLU(inplace=True),
        torch.nn.Linear(16, encoding_dim),
    ).to(device)

    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.eval()
    return encoder