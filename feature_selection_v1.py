from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import copy
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

from regressor_mini import RegressorMini


@dataclass
class TrainingConfig:
    """Configuration for model training with feature selection."""
    # Model hyperparameters
    dropout: float = 0
    learning_rate: float = 0.001
    weight_decay: float = 0
    batch_size: int = 128
    epochs: int = 100  # Reduced for feature selection speed

    # Data preprocessing
    clip_min: float = -7.0
    clip_max: float = 7.0
    seed: int = 42

    # Column groups
    impute_cols: List[str] = field(default_factory=lambda: [
        "length", "width", "rating", "item_weight", "price", "print_length", "height"
    ])
    transform_cols: List[str] = field(default_factory=lambda: [
        "q_since_first", "avg_discount_rate", "print_length", "length",
        "width", "height", "rating", "price", "item_weight"
    ])
    log_cols: List[str] = field(default_factory=lambda: [
        "q_since_first", "length", "width", "height", "price"
    ])
    clip_cols: List[str] = field(default_factory=lambda: [
        "avg_discount_rate", "rating", "item_weight", "q_since_first"
    ])
    dummy_cols: List[str] = field(default_factory=lambda: [
        "format", "channel", "q_num", "series"  # series is dummy, not meta
    ])
    meta_cols: List[str] = field(default_factory=lambda: [
        "isbn", "title", "year_quarter"
    ])
    target_col: str = "quantity"

    # Feature selection candidates
    potential_feat_cols: List[str] = field(default_factory=lambda: [
        "price", "height", "length", "width", "item_weight",
        "print_length", "rating", "q_since_first"
    ])


class DataPreprocessor:
    """Enhanced preprocessor with greedy feature selection."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scalers: Dict[str, RobustScaler] = {}
        self.y_scaler: RobustScaler = None
        self.series_medians: pd.DataFrame = None
        self.global_medians: pd.Series = None
        self.feature_columns: List[str] = None
        self.dummy_columns: List[str] = None
        self.best_features: List[str] = None  # From greedy selection

    def fit_imputation(self, df_train: pd.DataFrame) -> None:
        self.series_medians = df_train.groupby("series")[self.config.impute_cols].median()
        self.global_medians = df_train[self.config.impute_cols].median()
        print(f"Computed medians for {len(self.config.impute_cols)} columns")

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df_with_medians = df.join(self.series_medians, on="series", rsuffix="_median")

        for col in self.config.impute_cols:
            median_col = f"{col}_median"
            df[col] = df[col].fillna(df_with_medians[median_col]).fillna(self.global_medians[col])

        return df.drop(columns=[c for c in df.columns if c.endswith('_median')], errors='ignore')

    def fit_transform_features(self, df_train: pd.DataFrame) -> pd.DataFrame:
        df_train = df_train.copy()
        for col in self.config.transform_cols:
            values = df_train[col].to_numpy().astype(np.float32)

            if col in self.config.log_cols:
                values = np.log1p(values)

            scaler = RobustScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler

            if col in self.config.clip_cols:
                values = np.clip(values, self.config.clip_min, self.config.clip_max)

            df_train[col] = values

        print(f"Fitted {len(self.config.transform_cols)} feature scalers")
        return df_train

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.config.transform_cols:
            values = df[col].to_numpy().astype(np.float32)

            if col in self.config.log_cols:
                values = np.log1p(values)

            if col in self.scalers:
                values = self.scalers[col].transform(values.reshape(-1, 1)).flatten()

            if col in self.config.clip_cols:
                values = np.clip(values, self.config.clip_min, self.config.clip_max)

            df[col] = values
        return df

    def fit_transform_target(self, df_train: pd.DataFrame) -> np.ndarray:
        self.y_scaler = RobustScaler()
        target_log = np.log1p(df_train[self.config.target_col].to_numpy().reshape(-1, 1))
        transformed = self.y_scaler.fit_transform(target_log).flatten()
        print(f"Target scaler: center={self.y_scaler.center_[0]:.4f}, scale={self.y_scaler.scale_[0]:.4f}")
        return transformed

    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        target_log = np.log1p(df[self.config.target_col].to_numpy().reshape(-1, 1))
        return self.y_scaler.transform(target_log).flatten()

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        y_log_scaled = self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return np.expm1(y_log_scaled)

    def safe_create_dummies(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None,
                            df_test: pd.DataFrame = None) -> Tuple:
        train_dummy = pd.get_dummies(df_train, columns=self.config.dummy_cols, drop_first=True, dtype=int)
        self.dummy_columns = [c for c in train_dummy.columns
                              if any(c.startswith(prefix + "_") for prefix in self.config.dummy_cols)]

        print(f"Safe dummies: {len(self.dummy_columns)} columns")
        result = [train_dummy]

        if df_val is not None:
            val_dummy = pd.get_dummies(df_val, columns=self.config.dummy_cols, drop_first=True, dtype=int)
            result.append(val_dummy.reindex(columns=train_dummy.columns, fill_value=0))
        if df_test is not None:
            test_dummy = pd.get_dummies(df_test, columns=self.config.dummy_cols, drop_first=True, dtype=int)
            result.append(test_dummy.reindex(columns=train_dummy.columns, fill_value=0))

        return tuple(result)

    def safe_transform_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dummy_columns is None:
            raise ValueError("Run safe_create_dummies first")
        df_dummy = pd.get_dummies(df, columns=self.config.dummy_cols, drop_first=True, dtype=int)
        return df_dummy.reindex(columns=self.dummy_columns, fill_value=0)

    def greedy_forward_selection(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                                 y_train: torch.Tensor, y_val: torch.Tensor,
                                 device: torch.device) -> List[str]:
        """Greedy forward feature selection - YOUR LOGIC PRESERVED."""
        selected = []
        remaining = list(self.config.potential_feat_cols)
        best_rmse_history = []
        all_cols = [c for c in df_train.columns
                    if c not in self.config.meta_cols + [self.config.target_col]]

        print(f"\nPHASE 1: Greedy Forward Selection ({len(remaining)} candidates)")
        print("=" * 60)

        while remaining:
            best_rmse = float('inf')
            best_feature = None

            for feat in remaining:
                candidate = selected + [feat]
                drop_cols = [c for c in self.config.potential_feat_cols if c not in candidate]
                feat_cols = [c for c in all_cols if c not in drop_cols]

                if not feat_cols:
                    continue

                X_train_cand = torch.from_numpy(df_train[feat_cols].to_numpy(np.float32)).to(device)
                X_val_cand = torch.from_numpy(df_val[feat_cols].to_numpy(np.float32)).to(device)

                rmse = self._evaluate_feature_set(X_train_cand, y_train, X_val_cand, y_val, device)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_feature = feat

            if best_feature is None:
                break

            selected.append(best_feature)
            remaining.remove(best_feature)
            best_rmse_history.append((selected.copy(), best_rmse))
            print(f"'{best_feature}' → RMSE: {best_rmse:.4f} | Features: {len(selected)}")

        self.best_features = min(best_rmse_history, key=lambda x: x[1])[0]
        drop_cols = [c for c in self.config.potential_feat_cols if c not in self.best_features]
        print(f"\nBEST: {self.best_features}")
        print(f"DROP: {drop_cols}")
        return self.best_features, drop_cols

    def _evaluate_feature_set(self, X_train: torch.Tensor, y_train: torch.Tensor,
                              X_val: torch.Tensor, y_val: torch.Tensor, device: torch.device) -> float:
        """Fast model evaluation for feature selection (30 epochs)."""
        model = RegressorMini(input_dim=X_train.shape[1], dropout=self.config.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
        val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=128)

        for _ in range(30):  # Fast evaluation
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = sum(loss_fn(model(xb.to(device)), yb.to(device)).item() * len(xb)
                               for xb, yb in val_dl) / len(val_dl.dataset)

        return np.sqrt(val_loss)

    def get_final_features(self) -> List[str]:
        """Get final feature columns after selection."""
        if self.best_features is None:
            raise ValueError("Run greedy_forward_selection first")

        drop_cols = [c for c in self.config.potential_feat_cols if c not in self.best_features]
        self.feature_columns = [c for c in self.dummy_columns + self.best_features
                                if c not in self.config.meta_cols + drop_cols + [self.config.target_col]]
        return self.feature_columns


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model: RegressorMini = None
        self.best_state: dict = None
        self.train_history: Dict[str, List[float]] = {"train_rmse": [], "val_rmse": []}

    def _train_epoch(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module
    ) -> float:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        num_samples = 0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)
            num_samples += len(X_batch)

        return np.sqrt(total_loss / num_samples)

    def _validate(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module
    ) -> float:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                total_loss += loss.item() * len(X_batch)
                num_samples += len(X_batch)

        return np.sqrt(total_loss / num_samples)

    def train(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_val: torch.Tensor,
            y_val: torch.Tensor
    ) -> Tuple[float, int]:
        """Train model and track best performance."""
        self.model = RegressorMini(input_dim=X_train.shape[1], dropout=self.config.dropout).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.MSELoss()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.config.batch_size, shuffle=False)

        best_val_rmse = float('inf')
        best_epoch = 0

        print(f"\n{'=' * 60}\nTraining for {self.config.epochs} epochs | {X_train.shape[1]} features\n{'=' * 60}")

        for epoch in range(self.config.epochs):
            train_rmse = self._train_epoch(self.model, train_loader, optimizer, criterion)
            val_rmse = self._validate(self.model, val_loader, criterion)

            self.train_history["train_rmse"].append(train_rmse)
            self.train_history["val_rmse"].append(val_rmse)

            is_best = val_rmse < best_val_rmse
            if is_best:
                best_val_rmse = val_rmse
                best_epoch = epoch + 1
                self.best_state = copy.deepcopy(self.model.state_dict())

            if (epoch + 1) % 10 == 0 or is_best:
                status = " BEST" if is_best else ""
                print(f"Epoch {epoch + 1:3d} | Train: {train_rmse:.4f} | Val: {val_rmse:.4f}{status}")

        return best_val_rmse, best_epoch

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluate best model on test set."""
        if self.best_state is None:
            raise ValueError("No trained model found. Call train() first.")

        self.model.load_state_dict(self.best_state)
        self.model.eval()

        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.config.batch_size, shuffle=False)
        test_rmse = self._validate(self.model, test_loader, nn.MSELoss())
        return test_rmse

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions with the best model."""
        if self.best_state is None:
            raise ValueError("No trained model found. Call train() first.")

        self.model.load_state_dict(self.best_state)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X.to(self.device)).cpu().numpy().flatten()

        return predictions

    def print_training_history(self):
        print("\n" + "=" * 70)
        print("TRAINING HISTORY - RMSE Losses")
        print("=" * 70)
        print(f"{'Epoch':<10} {'Train RMSE':<18} {'Val RMSE':<18} {'Status':<15}")
        print("-" * 70)

        for epoch, (train_rmse, val_rmse) in enumerate(
                zip(self.train_history["train_rmse"], self.train_history["val_rmse"]),
                start=1
        ):
            if epoch % 10 == 0 or epoch == len(self.train_history["train_rmse"]):
                status = "Best" if epoch == len(self.train_history["train_rmse"]) else ""
                print(f"{epoch:<10} {train_rmse:<18.4f} {val_rmse:<18.4f} {status:<15}")

        print("=" * 70)

    def print_predictions_vs_actual(self, y_actual: np.ndarray, y_predicted: np.ndarray, data_type: str = "Validation"):
        print(f"\n{'=' * 85}")
        print(f"{data_type.upper()} SET - Predicted vs Actual Values (First 20 Samples)")
        print(f"{'=' * 85}")
        print(f"{'Index':<8} {'Actual':<15} {'Predicted':<15} {'Error':<15} {'% Error':<15}")
        print("-" * 85)

        num_samples = min(20, len(y_actual))
        for i in range(num_samples):
            actual = y_actual[i]
            pred = y_predicted[i]
            error = pred - actual
            pct_error = (error / actual * 100) if actual != 0 else 0.0
            print(f"{i:<8} {actual:<15.4f} {pred:<15.4f} {error:<15.4f} {pct_error:<15.2f}%")

        if len(y_actual) > 20:
            print(f"... and {len(y_actual) - 20} more samples")

        mae = np.mean(np.abs(y_actual - y_predicted))
        rmse = np.sqrt(np.mean((y_actual - y_predicted) ** 2))
        print("-" * 85)
        print(f"{'SUMMARY':<8} MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        print("=" * 85)

    def plot_rmse_curve(self, best_epoch: int = None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_history["train_rmse"], label="Training RMSE", linewidth=2.5, color="#1f77b4")
        plt.plot(self.train_history["val_rmse"], label="Validation RMSE", linewidth=2.5, color="#ff7f0e")
        if best_epoch:
            plt.axvline(best_epoch - 1, color="red", linestyle="--", label=f"Best (ep {best_epoch})")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.title("Book Sales RegressorMini — RMSE Curve", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self, X_test: torch.Tensor, y_test: torch.Tensor, preprocessor):
        """Updated to use preprocessor directly - FIXED."""
        import matplotlib.pyplot as plt

        preds_scaled = self.predict(X_test)
        actual_scaled = y_test.cpu().numpy().flatten()

        # Use proper inverse transform
        pred_qty = preprocessor.inverse_transform_target(preds_scaled)
        actual_qty = preprocessor.inverse_transform_target(actual_scaled)

        # Plot original scale only (simpler, more relevant)
        coeffs = np.polyfit(pred_qty, actual_qty, 1)
        fit_line = np.poly1d(coeffs)
        x_range = np.linspace(pred_qty.min(), pred_qty.max(), 100)

        plt.figure(figsize=(12, 8))
        plt.scatter(pred_qty, actual_qty, alpha=0.6, s=30, c='#1f77b4', label='Test samples')
        plt.plot(x_range, fit_line(x_range), 'r--', linewidth=3,
                 label=f'Best fit (y={coeffs[0]:.2f}x+{coeffs[1]:.2f})')
        plt.plot(x_range, x_range, 'g-', linewidth=2, alpha=0.8, label='Perfect prediction')
        plt.xlabel('Predicted Quantity', fontsize=14)
        plt.ylabel('Actual Quantity', fontsize=14)
        plt.title('Test Set: Actual vs Predicted Quantity (Original Scale)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    config = TrainingConfig()
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup paths
    results_folder = Path("results")
    results_folder.mkdir(parents=True, exist_ok=True)
    data_path = Path("data/target_series_new_with_features2.csv")
    target_path = Path("data/target_books_new.csv")

    # Load & split
    df_full = pd.read_csv(data_path, dtype={"isbn": "string"})
    df_train, df_temp = train_test_split(df_full, test_size=0.2, random_state=config.seed, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=config.seed + 1, shuffle=True)

    print(f"Dataset split: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    # Preprocessing pipeline
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    preprocessor = DataPreprocessor(config)
    preprocessor.fit_imputation(df_train)

    # Imputation
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        locals()[name.lower()] = preprocessor.impute_missing_values(df)

    # Feature transformation
    df_train = preprocessor.fit_transform_features(df_train)
    df_val = preprocessor.transform_features(df_val)
    df_test = preprocessor.transform_features(df_test)

    # Target transformation
    y_train_scaled = preprocessor.fit_transform_target(df_train)
    y_val_scaled = preprocessor.transform_target(df_val)
    y_test_scaled = preprocessor.transform_target(df_test)

    # Safe dummy encoding
    df_train, df_val, df_test = preprocessor.safe_create_dummies(df_train, df_val, df_test)

    # Tensors for feature selection
    y_train_t = torch.from_numpy(y_train_scaled.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val_scaled.astype(np.float32)).to(device)

    # FEATURE SELECTION - YOUR LOGIC INTEGRATED
    print("\n" + "=" * 60)
    best_features, drop_cols = preprocessor.greedy_forward_selection(
        df_train, df_val, y_train_t, y_val_t, device
    )
    feature_cols = preprocessor.get_final_features()
    print(f"\nFinal features: {len(feature_cols)} total ({len(best_features)} selected + dummies)")

    # Prepare final tensors
    X_train = torch.from_numpy(df_train[feature_cols].to_numpy(np.float32)).to(device)
    X_val = torch.from_numpy(df_val[feature_cols].to_numpy(np.float32)).to(device)
    X_test = torch.from_numpy(df_test[feature_cols].to_numpy(np.float32)).to(device)
    y_test_t = torch.from_numpy(y_test_scaled.astype(np.float32)).to(device)

    print(f"Final shapes: X_train={X_train.shape}, features={len(feature_cols)}")

    # Train final model
    trainer = ModelTrainer(config, device)
    best_val_rmse, best_epoch = trainer.train(X_train, y_train_t, X_val, y_val_t)

    # Save & evaluate
    torch.save(trainer.best_state, results_folder / "regressor_best_selected.pth")
    test_rmse = trainer.evaluate(X_test, y_test_t)
    print(f"\nBest Val RMSE: {best_val_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    # Final predictions
    print("\nGENERATING PRODUCTION PREDICTIONS")
    df_target = pd.read_csv(target_path, dtype={"isbn": "string"})
    final_ids = df_target[["isbn", "title"]].copy()

    df_target = preprocessor.impute_missing_values(df_target)
    df_target = preprocessor.transform_features(df_target)
    df_target = preprocessor.safe_transform_dummies(df_target)
    df_target = df_target.reindex(columns=feature_cols, fill_value=0.0)

    X_target = torch.from_numpy(df_target[feature_cols].to_numpy(np.float32)).to(device)
    pred_scaled = trainer.predict(X_target)
    pred_qty = np.clip(preprocessor.inverse_transform_target(pred_scaled), 0, None).round().astype(int)

    final_ids["pred_quantity"] = pred_qty
    final_ids.to_csv(results_folder / "final_predictions_selected.csv", index=False)
    print(f"Predictions saved: {results_folder / 'final_predictions_selected.csv'}")
    print(final_ids.head(10))

    # Diagnostics
    trainer.print_training_history()
    trainer.plot_rmse_curve(best_epoch)
    trainer.plot_actual_vs_predicted(X_test, y_test_t, preprocessor)


if __name__ == "__main__":
    main()
