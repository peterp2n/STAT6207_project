from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import copy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from regressor_mini import Regressor


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model hyperparameters
    dropout: float = 0
    learning_rate: float = 0.0001
    weight_decay: float = 0
    batch_size: int = 128
    epochs: int = 150

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
        "avg_discount_rate", "rating", "item_weight"
    ])

    dummy_cols: List[str] = field(default_factory=lambda: [
        "format", "channel", "q_num", "series"  # series is dummy, not meta
    ])

    meta_cols: List[str] = field(default_factory=lambda: [
        "isbn", "title", "year_quarter"
    ])

    drop_cols: List[str] = field(default_factory=lambda: [
        "number_of_reviews", "price", "height", "length", "width"
    ])

    target_col: str = "quantity"


class DataPreprocessor:
    """Handles all data preprocessing operations."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scalers: Dict[str, RobustScaler] = {}
        self.y_scaler: RobustScaler = None
        self.series_medians: pd.DataFrame = None
        self.global_medians: pd.Series = None
        self.feature_columns: List[str] = None
        self.dummy_columns: List[str] = None  # NEW: Track safe dummy columns

    def fit_imputation(self, df_train: pd.DataFrame) -> None:
        """Compute medians for imputation from training data."""
        self.series_medians = df_train.groupby("series")[self.config.impute_cols].median()
        self.global_medians = df_train[self.config.impute_cols].median()
        print(f"Computed medians for {len(self.config.impute_cols)} columns")

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using series-specific and global medians."""
        df = df.copy()  # Safe copy
        df_with_medians = df.join(self.series_medians, on="series", rsuffix="_median")

        for col in self.config.impute_cols:
            median_col = f"{col}_median"
            df[col] = df[col].fillna(df_with_medians[median_col]).fillna(self.global_medians[col])

        return df.drop(columns=[c for c in df.columns if c.endswith('_median')], errors='ignore')

    def fit_transform_features(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers and transform training features."""
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

        print(f"Fitted and transformed {len(self.config.transform_cols)} feature columns")
        return df_train

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers."""
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
        """Fit scaler and transform training target."""
        self.y_scaler = RobustScaler()
        target_log = np.log1p(df_train[self.config.target_col].to_numpy().reshape(-1, 1))
        transformed = self.y_scaler.fit_transform(target_log).flatten()
        print(f"Fitted target scaler: center={self.y_scaler.center_[0]:.4f}, scale={self.y_scaler.scale_[0]:.4f}")
        return transformed

    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        """Transform target using fitted scaler."""
        target_log = np.log1p(df[self.config.target_col].to_numpy().reshape(-1, 1))
        return self.y_scaler.transform(target_log).flatten()

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale - FIXED."""
        y_log_scaled = self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return np.expm1(y_log_scaled)

    def safe_create_dummies(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None,
                            df_test: pd.DataFrame = None) -> Tuple:
        """Safe dummy encoding - fit on train, apply to all splits - NEW."""
        # Fit dummies on train only
        train_dummy = pd.get_dummies(df_train, columns=self.config.dummy_cols, drop_first=True, dtype=int)

        # Extract dummy column names from train
        self.dummy_columns = [c for c in train_dummy.columns
                              if any(c.startswith(prefix + "_") for prefix in self.config.dummy_cols)]

        print(f"Safe dummies created: {len(self.dummy_columns)} columns from {self.config.dummy_cols}")

        # Reindex val/test to match train exactly
        result = [train_dummy]

        if df_val is not None:
            val_dummy = pd.get_dummies(df_val, columns=self.config.dummy_cols, drop_first=True, dtype=int)
            val_dummy = val_dummy.reindex(columns=train_dummy.columns, fill_value=0)
            result.append(val_dummy)

        if df_test is not None:
            test_dummy = pd.get_dummies(df_test, columns=self.config.dummy_cols, drop_first=True, dtype=int)
            test_dummy = test_dummy.reindex(columns=train_dummy.columns, fill_value=0)
            result.append(test_dummy)

        return tuple(result)

    def safe_transform_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data with safe dummy columns - for target books."""
        if self.dummy_columns is None:
            raise ValueError("Must call safe_create_dummies() first")

        df_dummy = pd.get_dummies(df, columns=self.config.dummy_cols, drop_first=True, dtype=int)
        return df_dummy.reindex(columns=self.dummy_columns, fill_value=0)

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select final features for model training."""
        if self.feature_columns is None:
            self.feature_columns = [
                c for c in df.columns
                if c not in self.config.meta_cols + self.config.drop_cols + [self.config.target_col]
            ]
            print(f"\nSelected {len(self.feature_columns)} features")
            print(f"Using first 10: {self.feature_columns[:10]}...")
        return df[self.feature_columns], self.feature_columns


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model: Regressor = None
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
        self.model = Regressor(input_dim=X_train.shape[1], dropout=self.config.dropout).to(self.device)
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
                status = " ★ BEST" if is_best else ""
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

    # [Keep all your existing print/plot methods unchanged - they work perfectly]
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
        plt.title("Book Sales Regressor — RMSE Curve", fontsize=14)
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


def setup_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_and_split_data(data_path: Path, config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n{'=' * 60}\nLoading data from {data_path}\n{'=' * 60}")

    dtype_spec = {
        "isbn": "string",
        **{col: "float32" for col in config.impute_cols + ["number_of_reviews", "avg_discount_rate"]}
    }

    df_full = pd.read_csv(data_path, dtype=dtype_spec)
    print(f"Loaded {len(df_full)} records")

    df_train, df_temp = train_test_split(df_full, test_size=0.2, random_state=config.seed, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=config.seed + 1, shuffle=True)

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    return df_train, df_val, df_test


def prepare_tensors(df: pd.DataFrame, feature_cols: List[str], target_values: np.ndarray, device: torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(df[feature_cols].to_numpy().astype(np.float32))
    y = torch.from_numpy(target_values.astype(np.float32))
    return X, y


def main():
    config = TrainingConfig()
    device = setup_device()
    print(f"Using device: {device}")
    set_random_seeds(config.seed)

    # Consistent paths
    results_folder = Path("results")
    results_folder.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/target_series_new_with_features2.csv")
    target_books_path = Path("data/target_books_new.csv")

    # Load and split
    df_train, df_val, df_test = load_and_split_data(data_path, config)

    # Preprocess
    print("\n" + "=" * 60)
    print("Preprocessing pipeline")
    print("=" * 60)

    preprocessor = DataPreprocessor(config)
    preprocessor.fit_imputation(df_train)

    df_train = preprocessor.impute_missing_values(df_train)
    df_val = preprocessor.impute_missing_values(df_val)
    df_test = preprocessor.impute_missing_values(df_test)

    df_train = preprocessor.fit_transform_features(df_train)
    df_val = preprocessor.transform_features(df_val)
    df_test = preprocessor.transform_features(df_test)

    y_train_scaled = preprocessor.fit_transform_target(df_train)
    y_val_scaled = preprocessor.transform_target(df_val)
    y_test_scaled = preprocessor.transform_target(df_test)

    # SAFE DUMMY ENCODING - FIXED
    df_train, df_val, df_test = preprocessor.safe_create_dummies(df_train, df_val, df_test)

    # Select features
    df_train_selected, feature_cols = preprocessor.select_features(df_train)
    df_val_selected = df_val[feature_cols]
    df_test_selected = df_test[feature_cols]

    # Tensors
    X_train, y_train = prepare_tensors(df_train_selected, feature_cols, y_train_scaled, device)
    X_val, y_val = prepare_tensors(df_val_selected, feature_cols, y_val_scaled, device)
    X_test, y_test = prepare_tensors(df_test_selected, feature_cols, y_test_scaled, device)

    print(f"Final feature count: {len(feature_cols)}")

    # Train
    trainer = ModelTrainer(config, device)
    best_val_rmse, best_epoch = trainer.train(X_train, y_train, X_val, y_val)

    print(f"\nBest validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")

    # Save model
    torch.save(trainer.best_state, results_folder / "regressor_best.pth")
    print(f"Model saved to results/regressor_best.pth")

    # Test evaluation
    test_rmse_scaled = trainer.evaluate(X_test, y_test)
    print(f"Test RMSE (scaled): {test_rmse_scaled:.4f}")

    # Diagnostics
    trainer.print_training_history()

    # Original scale evaluation
    y_val_pred = trainer.predict(X_val)
    y_test_pred = trainer.predict(X_test)

    y_val_orig = preprocessor.inverse_transform_target(y_val.cpu().numpy())
    y_val_pred_orig = preprocessor.inverse_transform_target(y_val_pred)
    y_test_orig = preprocessor.inverse_transform_target(y_test.cpu().numpy())
    y_test_pred_orig = preprocessor.inverse_transform_target(y_test_pred)

    trainer.print_predictions_vs_actual(y_val_orig, y_val_pred_orig, "Validation")
    trainer.print_predictions_vs_actual(y_test_orig, y_test_pred_orig, "Test")

    print("\nPlotting results...")
    trainer.plot_rmse_curve(best_epoch)
    trainer.plot_actual_vs_predicted(X_test, y_test, preprocessor)

    # MOVED TO END: Final target predictions
    print("\n" + "=" * 60)
    print("FINAL TARGET BOOK PREDICTIONS")
    print("=" * 60)

    df_target = pd.read_csv(target_books_path, dtype={"isbn": "string"})
    final_ids = df_target[["isbn", "title"]].copy()

    df_target = preprocessor.impute_missing_values(df_target)
    df_target = preprocessor.transform_features(df_target)
    df_target = preprocessor.safe_transform_dummies(df_target)  # SAFE
    df_target = df_target.reindex(columns=feature_cols, fill_value=0.0)

    X_target = torch.from_numpy(df_target[feature_cols].to_numpy().astype(np.float32))
    predictions_scaled = trainer.predict(X_target)
    predictions = np.clip(preprocessor.inverse_transform_target(predictions_scaled), 0, None).round().astype(int)

    final_ids["pred_quantity"] = predictions
    final_ids.to_csv(results_folder / "final_predictions.csv", index=False)
    print("Predictions saved to results/final_predictions.csv")
    print("\nFINAL TARGET PREDICTIONS (first 10):")
    print(final_ids.head(10))
    print(f"\nTotal predictions generated: {len(final_ids)} books")
    print("=" * 60)

    print(f"\nPipeline complete! Test RMSE (scaled): {test_rmse_scaled:.4f}")


if __name__ == "__main__":
    main()
