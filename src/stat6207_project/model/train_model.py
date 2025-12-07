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

from regressor import Regressor


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model hyperparameters
    dropout: float = 0.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 100

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
        "format", "channel", "q_num"
    ])

    meta_cols: List[str] = field(default_factory=lambda: [
        "isbn", "title", "year_quarter", "series"
    ])

    target_col: str = "quantity"

    features_to_use: List[str] = field(default_factory=lambda: [
        "q_since_first",
        "avg_discount_rate"
        "price", "height", "length", "width", "item_weight",
        "print_length", "rating",
    ])


class DataPreprocessor:
    """Handles all data preprocessing operations."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scalers: Dict[str, RobustScaler] = {}
        self.y_scaler: RobustScaler = None
        self.series_medians: pd.DataFrame = None
        self.global_medians: pd.Series = None
        self.feature_columns: List[str] = None

    def fit_imputation(self, df_train: pd.DataFrame) -> None:
        """Compute medians for imputation from training data."""
        self.series_medians = df_train.groupby("series")[self.config.impute_cols].median()
        self.global_medians = df_train[self.config.impute_cols].median()
        print(f"Computed medians for {len(self.config.impute_cols)} columns")

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using series-specific and global medians."""
        df_with_medians = df.join(self.series_medians, on="series", rsuffix="_median")

        for col in self.config.impute_cols:
            median_col = f"{col}_median"
            df[col] = df[col].fillna(df_with_medians[median_col]).fillna(self.global_medians[col])

        return df

    def fit_transform_features(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers and transform training features."""
        for col in self.config.transform_cols:
            values = df_train[col].to_numpy().astype(np.float32)

            # Apply log transformation if needed
            if col in self.config.log_cols:
                values = np.log1p(values)

            # Fit and transform with RobustScaler
            scaler = RobustScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler

            # Apply clipping if needed
            if col in self.config.clip_cols:
                values = np.clip(values, self.config.clip_min, self.config.clip_max)

            df_train[col] = values

        print(f"Fitted and transformed {len(self.config.transform_cols)} feature columns")
        return df_train

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers."""
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
        """Convert scaled predictions back to original scale."""
        y_log = y_scaled * self.y_scaler.scale_[0] + self.y_scaler.center_[0]
        return np.expm1(y_log)

    def create_dummy_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical columns."""
        return pd.get_dummies(df, columns=self.config.dummy_cols, drop_first=True)

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select final features for model training."""
        if self.feature_columns is None:
            drop_cols = [c for c in self.config.transform_cols if c not in self.config.features_to_use]
            self.feature_columns = [
                c for c in df.columns
                if c not in self.config.meta_cols + drop_cols + [self.config.target_col]
            ]
            print(f"\nSelected {len(self.feature_columns)} features")
            print(f"Using: {self.config.features_to_use}")
            print(f"Dropped: {drop_cols}")

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
        # Initialize model and optimizer
        self.model = Regressor(input_dim=X_train.shape[1], dropout=self.config.dropout).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.MSELoss()

        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Training loop
        best_val_rmse = float('inf')
        best_epoch = 0

        print(f"\n{'=' * 60}\nTraining for {self.config.epochs} epochs\n{'=' * 60}")

        for epoch in range(self.config.epochs):
            train_rmse = self._train_epoch(self.model, train_loader, optimizer, criterion)
            val_rmse = self._validate(self.model, val_loader, criterion)

            self.train_history["train_rmse"].append(train_rmse)
            self.train_history["val_rmse"].append(val_rmse)

            # Track best model
            is_best = val_rmse < best_val_rmse
            if is_best:
                best_val_rmse = val_rmse
                best_epoch = epoch + 1
                self.best_state = copy.deepcopy(self.model.state_dict())

            # Print progress
            if (epoch + 1) % 10 == 0 or is_best:
                status = " ★ BEST" if is_best else ""
                print(f"Epoch {epoch + 1:3d} | Train: {train_rmse:.4f} | Val: {val_rmse:.4f}{status}")

        return best_val_rmse, best_epoch

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluate best model on test set."""
        if self.best_state is None:
            raise ValueError("No trained model found. Call train() first.")

        # Load best model
        self.model.load_state_dict(self.best_state)
        self.model.eval()

        # Evaluate
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.config.batch_size,
            shuffle=False
        )
        test_rmse = self._validate(self.model, test_loader, nn.MSELoss())

        return test_rmse

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions with the best model."""
        if self.best_state is None:
            raise ValueError("No trained model found. Call train() first.")

        self.model.load_state_dict(self.best_state)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X).cpu().numpy().flatten()

        return predictions


def setup_device() -> torch.device:
    """Determine and return the best available device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_and_split_data(
        data_path: Path,
        config: TrainingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and split into train/val/test sets."""
    print(f"\n{'=' * 60}\nLoading data from {data_path}\n{'=' * 60}")

    dtype_spec = {
        "isbn": "string",
        **{col: "float32" for col in config.impute_cols + ["number_of_reviews", "avg_discount_rate"]}
    }

    df_full = pd.read_csv(data_path, dtype=dtype_spec)
    print(f"Loaded {len(df_full)} records")

    # Split data
    df_train, df_temp = train_test_split(
        df_full, test_size=0.2, random_state=config.seed, shuffle=True
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=config.seed + 1, shuffle=True
    )

    print(f"Split: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")
    return df_train, df_val, df_test


def prepare_tensors(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_values: np.ndarray,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dataframe to PyTorch tensors."""
    X = torch.from_numpy(df[feature_cols].to_numpy().astype(np.float32)).to(device)
    y = torch.from_numpy(target_values.astype(np.float32)).to(device)
    return X, y


def main():
    """Main training pipeline."""
    # Setup
    config = TrainingConfig()
    device = setup_device()
    set_random_seeds(config.seed)

    # Create output directory
    results_folder = Path("results")
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load and split data
    data_path = Path("data") / "target_series_new_with_features2.csv"
    df_train, df_val, df_test = load_and_split_data(data_path, config)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)

    # Fit preprocessing on training data
    print("\n" + "=" * 60)
    print("Preprocessing data")
    print("=" * 60)

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

    # Create dummy variables
    df_train = preprocessor.create_dummy_variables(df_train)
    df_val = preprocessor.create_dummy_variables(df_val)
    df_test = preprocessor.create_dummy_variables(df_test)

    # Select features
    _, feature_cols = preprocessor.select_features(df_train)

    # Prepare tensors
    X_train, y_train = prepare_tensors(df_train, feature_cols, y_train_scaled, device)
    X_val, y_val = prepare_tensors(df_val, feature_cols, y_val_scaled, device)
    X_test, y_test = prepare_tensors(df_test, feature_cols, y_test_scaled, device)

    # Train model
    trainer = ModelTrainer(config, device)
    best_val_rmse, best_epoch = trainer.train(X_train, y_train, X_val, y_val)

    print(f"\n{'=' * 60}")
    print(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
    print(f"{'=' * 60}")

    # Save model
    model_path = results_folder / "regressor_best.pth"
    torch.save(trainer.best_state, model_path)
    print(f"\nModel saved to {model_path}")

    # Evaluate on test set
    test_rmse = trainer.evaluate(X_test, y_test)
    print(f"\n{'=' * 60}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"{'=' * 60}")

    # Generate final predictions
    print("\n" + "=" * 60)
    print("Generating final predictions")
    print("=" * 60)

    target_books_path = Path("data") / "target_books_new.csv"
    df_target = pd.read_csv(target_books_path, dtype={"isbn": "string"})
    final_ids = df_target[["isbn", "title"]].copy()

    # Preprocess target books
    df_target = preprocessor.impute_missing_values(df_target)
    df_target = preprocessor.transform_features(df_target)
    df_target = preprocessor.create_dummy_variables(df_target)

    # Ensure all feature columns exist
    df_target = df_target.reindex(columns=feature_cols, fill_value=0.0)

    # Prepare tensor and predict
    X_target = torch.from_numpy(df_target.to_numpy().astype(np.float32)).to(device)
    predictions_scaled = trainer.predict(X_target)

    # Inverse transform predictions
    predictions = preprocessor.inverse_transform_target(predictions_scaled)
    final_ids["pred_quantity"] = predictions.round(0).astype(int)

    # Save predictions
    output_path = results_folder / "final_predictions.csv"
    final_ids.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(f"\nSample predictions:\n{final_ids.head(10)}")


if __name__ == "__main__":
    main()