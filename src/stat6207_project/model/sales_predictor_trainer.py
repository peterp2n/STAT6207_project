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
        self.lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

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
        best_epoch = 0
        wait = 0
        best_filepath = None  # Will hold Path to the current best model file

        # Make sure folder exists
        save_dir = Path("sales_results")
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % print_every == 0 or epoch == epochs:
                print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

            # === Save only the SINGLE best model (with dynamic name) ===
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_epoch = epoch
                wait = 0

                # Build dynamic filename
                new_filepath = save_dir / f"sales_predictor_best_epoch_{epoch}_lr_{self.lr:.0e}_bs_{train_loader.batch_size}.pth"

                # Delete previous best file if exists
                if best_filepath is not None and best_filepath.exists():
                    best_filepath.unlink()
                    print(f"  (Replaced old best: {best_filepath.name})")

                # Save new best
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val,
                    'lr': self.lr,
                    'batch_size': train_loader.batch_size,
                }, new_filepath)

                best_filepath = new_filepath
                print(f"  New best model saved: {new_filepath.name} | Val MSE: {best_val:.6f}")

            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best model was at epoch {best_epoch} with Val MSE: {best_val:.6f}")
                    break

        # === Load the best (and only) saved model ===
        if best_filepath and best_filepath.exists():
            checkpoint = torch.load(best_filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nBest model loaded: {best_filepath.name}")
            print(f"   Epoch {checkpoint['epoch']} | Val MSE: {checkpoint['val_loss']:.6f}")
        else:
            print("\nNo best model found — using final weights.")

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