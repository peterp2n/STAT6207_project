# sales_predictor_trainer_sim.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from sales_predictor_sim import SalesPredictor


class SalesPredictorTrainerSim:
    def __init__(
        self,
        encoded_dim: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
        use_layernorm: bool = True,
        leaky_slope: float = 0.01
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.model = SalesPredictor(
            encoded_dim=encoded_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
            leaky_slope=leaky_slope
        ).to(self.device)

        # Losses
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()   # MAE

        self.encoded_dim = encoded_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.leaky_slope = leaky_slope

        self.lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

        # Store all metrics
        self.train_losses = []
        self.val_losses   = []
        self.train_mae    = []
        self.val_mae      = []
        self.train_rmse   = []
        self.val_rmse     = []

    def _compute_metrics(self, loader: DataLoader, is_train: bool = False):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_mse = total_mae = total_samples = 0.0
        with torch.no_grad() if not is_train else torch.enable_grad():
            for enc, y in loader:
                enc, y = enc.to(self.device), y.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()
                    pred = self.model(enc)
                    loss_mse = self.criterion_mse(pred, y)
                    loss_mse.backward()
                    self.optimizer.step()
                else:
                    pred = self.model(enc)

                batch_size = y.size(0)
                total_mse += self.criterion_mse(pred, y).item() * batch_size
                total_mae += self.criterion_mae(pred, y).item() * batch_size
                total_samples += batch_size

        mse = total_mse / total_samples
        mae = total_mae / total_samples
        rmse = np.sqrt(mse)

        return mse, mae, rmse

    def _val_epoch(self, loader: DataLoader):
        mse, mae, rmse = self._compute_metrics(loader, is_train=False)
        return mse, mae, rmse

    def _train_epoch(self, loader: DataLoader):
        mse, mae, rmse = self._compute_metrics(loader, is_train=True)
        return mse, mae, rmse

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 1000,
            patience: int = 30,
            print_every: int = 10
    ):
        best_val_mse = float('inf')
        best_epoch = 0
        wait = 0
        best_filepath = None
        save_dir = Path("sales_results_sim")
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_mse, train_mae, train_rmse = self._train_epoch(train_loader)
            val_mse,   val_mae,   val_rmse   = self._val_epoch(val_loader)

            # Store
            self.train_losses.append(train_mse)
            self.val_losses.append(val_mse)
            self.train_mae.append(train_mae)
            self.val_mae.append(val_mae)
            self.train_rmse.append(train_rmse)
            self.val_rmse.append(val_rmse)

            if epoch % print_every == 0 or epoch == epochs:
                print(f"Epoch {epoch:3d} | "
                      f"Train MSE: {train_mse:.5f} MAE: {train_mae:.4f} RMSE: {train_rmse:.4f} | "
                      f"Val   MSE: {val_mse:.5f} MAE: {val_mae:.4f} RMSE: {val_rmse:.4f}")

            # Early stopping + best model saving on VAL MSE (you can switch to RMSE if you prefer)
            if val_mse < best_val_mse - 1e-5:
                best_val_mse = val_mse
                best_epoch = epoch
                wait = 0

                new_filepath = save_dir / f"sales_predictor_sim_best_epoch_{epoch}_lr_{self.lr:.0e}_bs_{train_loader.batch_size}.pth"
                if best_filepath and best_filepath.exists():
                    best_filepath.unlink()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_mse': best_val_mse,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'lr': self.lr,
                    'batch_size': train_loader.batch_size,

                    'encoded_dim': self.encoded_dim,
                    'dropout': self.dropout,
                    'use_layernorm': self.use_layernorm,
                    'leaky_slope': self.leaky_slope,
                }, new_filepath)

                best_filepath = new_filepath
                print(f"  → New best (Val MSE {best_val_mse:.5f} | MAE {val_mae:.4f} | RMSE {val_rmse:.4f})")

            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best epoch: {best_epoch} → Val MSE: {best_val_mse:.5f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")
                    break

        if best_filepath and best_filepath.exists():
            ckpt = torch.load(best_filepath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"\nBest model loaded from epoch {ckpt['epoch']} | "
                  f"Val MSE: {ckpt['val_mse']:.5f} | MAE: {ckpt.get('val_mae', 'N/A'):.4f} | RMSE: {np.sqrt(ckpt['val_mse']):.4f}")

    def plot_losses(self, save_path: Optional[Path] = None):
        epochs = range(1, len(self.train_losses) + 1)

        # MSE figure
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Train MSE")
        plt.plot(epochs, self.val_losses, label="Val MSE", linestyle="--")
        plt.yscale('log')
        plt.title("MSE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            mse_path = save_path.parent / f"{save_path.stem}_mse{save_path.suffix}"
            plt.savefig(mse_path, dpi=300, bbox_inches='tight')
            print(f"MSE plot saved to {mse_path}")
        plt.show()

        # MAE figure
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_mae, label="Train MAE")
        plt.plot(epochs, self.val_mae, label="Val MAE", linestyle="--")
        plt.title("MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            mae_path = save_path.parent / f"{save_path.stem}_mae{save_path.suffix}"
            plt.savefig(mae_path, dpi=300, bbox_inches='tight')
            print(f"MAE plot saved to {mae_path}")
        plt.show()

        # RMSE figure
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_rmse, label="Train RMSE")
        plt.plot(epochs, self.val_rmse, label="Val RMSE", linestyle="--")
        plt.title("RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            rmse_path = save_path.parent / f"{save_path.stem}_rmse{save_path.suffix}"
            plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
            print(f"RMSE plot saved to {rmse_path}")
        plt.show()