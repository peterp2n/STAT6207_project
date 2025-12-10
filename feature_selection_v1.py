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

from regressor_mini import Regressor


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
        print(f"✅ Computed medians for {len(self.config.impute_cols)} columns")

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

        print(f"✅ Fitted {len(self.config.transform_cols)} feature scalers")
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
        print(f"✅ Target scaler: center={self.y_scaler.center_[0]:.4f}, scale={self.y_scaler.scale_[0]:.4f}")
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

        print(f"✅ Safe dummies: {len(self.dummy_columns)} columns")
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

        print(f"\n🚀 PHASE 1: Greedy Forward Selection ({len(remaining)} candidates)")
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
            print(f"➕ '{best_feature}' → RMSE: {best_rmse:.4f} | Features: {len(selected)}")

        self.best_features = min(best_rmse_history, key=lambda x: x[1])[0]
        drop_cols = [c for c in self.config.potential_feat_cols if c not in self.best_features]
        print(f"\n🏆 BEST: {self.best_features}")
        print(f"🗑️  DROP: {drop_cols}")
        return self.best_features, drop_cols

    def _evaluate_feature_set(self, X_train: torch.Tensor, y_train: torch.Tensor,
                              X_val: torch.Tensor, y_val: torch.Tensor, device: torch.device) -> float:
        """Fast model evaluation for feature selection (30 epochs)."""
        model = Regressor(input_dim=X_train.shape[1], dropout=self.config.dropout).to(device)
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


# [ModelTrainer class remains exactly the same as your previous version]
class ModelTrainer:
    # ... [Keep all your existing ModelTrainer methods unchanged - they work perfectly]
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model: Regressor = None
        self.best_state: dict = None
        self.train_history: Dict[str, List[float]] = {"train_rmse": [], "val_rmse": []}

    # [Include all your existing _train_epoch, _validate, train, evaluate, predict,
    #  print_training_history, print_predictions_vs_actual, plot_rmse_curve, plot_actual_vs_predicted]
    # ... [exactly as in your previous OOP version]


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

    # 🔥 FEATURE SELECTION - YOUR LOGIC INTEGRATED
    print("\n" + "=" * 60)
    best_features, drop_cols = preprocessor.greedy_forward_selection(
        df_train, df_val, y_train_t, y_val_t, device
    )
    feature_cols = preprocessor.get_final_features()
    print(f"\n✅ Final features: {len(feature_cols)} total ({len(best_features)} selected + dummies)")

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
    print(f"\n🏆 Best Val RMSE: {best_val_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    # Final predictions
    print("\n🔮 GENERATING PRODUCTION PREDICTIONS")
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
    print(f"\n✅ Predictions saved: {results_folder / 'final_predictions_selected.csv'}")
    print(final_ids.head(10))

    # Diagnostics
    trainer.print_training_history()
    trainer.plot_rmse_curve(best_epoch)
    trainer.plot_actual_vs_predicted(X_test, y_test_t, preprocessor)


if __name__ == "__main__":
    main()
