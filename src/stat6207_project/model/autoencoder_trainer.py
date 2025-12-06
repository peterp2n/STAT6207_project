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
        self.model = AutoEncoder(input_dim, encoding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

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
                inputs, targets = batch  # Already on correct device
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(loader)
            self.train_losses.append(avg_train_loss)

            val_mse = None
            if val_data is not None:
                val_mse = self._compute_mse(val_data)
                self.val_losses.append(val_mse)
            else:
                self.val_losses.append(None)

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
        test_data = test_data.to(self.device)
        return self._compute_mse(test_data)

    def get_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(data)
        self.model.train()  # Restore training mode if needed
        return embeddings

    def plot_losses(self,
                    figsize: Tuple[int, int] = (10, 6),
                    title: str = "Autoencoder Training & Validation MSE",
                    save_path: Optional[str] = None,
                    y_scale: str = 'auto',  # ← now supports 'auto'
                    y_lim: Union[str, Tuple[float, float], None] = 'auto',
                    y_ticks: Optional[List[float]] = None,
                    y_tick_labels: Optional[List[str]] = None):

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=figsize)

        plt.plot(epochs, self.train_losses, label="Train MSE", color="tab:blue", linewidth=2.5)

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

        # === SMART DYNAMIC Y-SCALE ===
        if y_scale == 'auto' or y_lim == 'auto':
            train_losses_np = np.array(self.train_losses)
            min_loss = train_losses_np.min()
            max_loss = train_losses_np.max()
            loss_ratio = max_loss / max_loss  # just to avoid division by zero

            # If loss drops by more than 2 orders of magnitude → use log scale
            if max_loss / max(min_loss, 1e-8) > 100:  # > 2 orders of magnitude
                final_y_scale = 'log'
                final_y_lim = (max(min_loss * 0.5, 1e-8), max_loss * 2)
            else:
                final_y_scale = 'linear'
                final_y_lim = (0, max_loss * 1.1)

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