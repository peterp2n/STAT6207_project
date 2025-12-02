import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np

from sales_predictor import SalesPredictor


class SalesPredictorTrainer:
    def __init__(
        self,
        encoded_dim: int,
        text_dim: int,
        image_dim: int,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.2
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.model = SalesPredictor(
            encoded_dim=encoded_dim,
            text_dim=text_dim,
            image_dim=image_dim,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for enc, txt, img, y in loader:
            enc, txt, img, y = enc.to(self.device), txt.to(self.device), img.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(enc, txt, img)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
        return total_loss / len(loader.dataset)

    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for enc, txt, img, y in loader:
                enc, txt, img, y = enc.to(self.device), txt.to(self.device), img.to(self.device), y.to(self.device)
                pred = self.model(enc, txt, img)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * y.size(0)
        return total_loss / len(loader.dataset)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 500,
        patience: int = 30,
        print_every: int = 10
    ):
        best_val = float('inf')
        wait = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # prev_lr = self.optimizer.param_groups[0]['lr']
            # current_lr = self.optimizer.param_groups[0]['lr']
            # if current_lr < prev_lr:
            #     print(f"Epoch {epoch}: Reducing LR from {prev_lr:.2e} to {current_lr:.2e}")

            if epoch % print_every == 0 or epoch == epochs:
                print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

            # Early stopping + save best
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                wait = 0
                torch.save(self.model.state_dict(), "sales_results/sales_predictor_best.pth")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load("sales_results/sales_predictor_best.pth"))
        print(f"Best validation MSE: {best_val:.6f}")

    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for enc, txt, img, _ in loader:  # ignore y if present
                enc, txt, img = enc.to(self.device), txt.to(self.device), img.to(self.device)
                pred = self.model(enc, txt, img)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def plot_losses(self, save_path: Optional[Path] = None):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label="Train MSE", linewidth=2)
        plt.plot(epochs, self.val_losses, label="Val MSE", linewidth=2, linestyle="--")
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Sales Predictor Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curve saved to {save_path}")
        plt.show()