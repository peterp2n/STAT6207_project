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
from regressor_mini import RegressorMini
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model hyperparameters
    dropout: float = 0.3
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 250
    patience: int = 50

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
        "format", "channel", "q_num", "series"
    ])

    meta_cols: List[str] = field(default_factory=lambda: [
        "isbn", "title", "year_quarter"
    ])

    drop_cols: List[str] = field(default_factory=lambda: [
        "number_of_reviews", 'price', 'height', 'length', 'width', 'item_weight', 'print_length', 'rating'
    ])

    target_col: str = "quantity"


class DataPreprocessor:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scalers: Dict[str, RobustScaler] = {}
        self.y_scaler: RobustScaler = None
        self.series_medians: pd.DataFrame = None
        self.global_medians: pd.Series = None
        self.feature_columns: List[str] = None
        self.dummy_columns: List[str] = None

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
        print(f"Fitted and transformed {len(self.config.transform_cols)} feature columns")
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
        print(f"Fitted target scaler: center={self.y_scaler.center_[0]:.4f}, scale={self.y_scaler.scale_[0]:.4f}")
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
        print(f"Safe dummies created: {len(self.dummy_columns)} columns")

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
        if self.dummy_columns is None:
            raise ValueError("Must call safe_create_dummies() first")
        df_dummy = pd.get_dummies(df, columns=self.config.dummy_cols, drop_first=True, dtype=int)
        return df_dummy.reindex(columns=self.dummy_columns, fill_value=0)

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if self.feature_columns is None:
            self.feature_columns = [
                c for c in df.columns
                if c not in self.config.meta_cols + self.config.drop_cols + [self.config.target_col]
            ]
            print(f"\nSelected {len(self.feature_columns)} features")
            print(f"Using first 10: {self.feature_columns[:10]}...")
        return df[self.feature_columns], self.feature_columns


class ModelTrainer:
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model: RegressorMini = None
        self.best_state: dict = None
        self.train_history: Dict[str, List[float]] = {"train_rmse": [], "val_rmse": []}

    def _train_epoch(self, model, dataloader, optimizer, criterion) -> float:
        model.train()
        total_loss = num_samples = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
            num_samples += len(X_batch)
        return np.sqrt(total_loss / num_samples)

    def _validate(self, model, dataloader, criterion) -> float:
        model.eval()
        total_loss = num_samples = 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
                num_samples += len(X_batch)
        return np.sqrt(total_loss / num_samples)

    def train(self, X_train, y_train, X_val, y_val) -> Tuple[float, int]:
        self.model = RegressorMini(input_dim=X_train.shape[1], dropout=self.config.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        criterion = nn.MSELoss()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.config.batch_size, shuffle=False)

        best_val_rmse = float('inf')
        best_epoch = 0
        epochs_no_improve = 0

        print(f"\n{'=' * 60}")
        print(f"Training up to {self.config.epochs} epochs | Patience: {self.config.patience} | Features: {X_train.shape[1]}")
        print(f"{'=' * 60}")

        for epoch in range(self.config.epochs):
            train_rmse = self._train_epoch(self.model, train_loader, optimizer, criterion)
            val_rmse = self._validate(self.model, val_loader, criterion)

            self.train_history["train_rmse"].append(train_rmse)
            self.train_history["val_rmse"].append(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch + 1
                self.best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                status = " [BEST]"
            else:
                epochs_no_improve += 1
                status = f" [{epochs_no_improve}/{self.config.patience}]"

            if (epoch + 1) % 10 == 0 or status == " [BEST]":
                print(f"Epoch {epoch + 1:3d} | Train: {train_rmse:.4f} | Val: {val_rmse:.4f}{status}")

            if epochs_no_improve >= self.config.patience:
                print(f"\nEarly stopping triggered after {self.config.patience} epochs without improvement!")
                break
        else:
            print(f"\nCompleted all {self.config.epochs} epochs.")

        print(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
        return best_val_rmse, best_epoch

    def evaluate(self, X_test, y_test) -> float:
        if not self.best_state:
            raise ValueError("No trained model.")
        self.model.load_state_dict(self.best_state)
        return self._validate(self.model, DataLoader(TensorDataset(X_test, y_test), batch_size=self.config.batch_size), nn.MSELoss())

    def predict(self, X: torch.Tensor) -> np.ndarray:
        if not self.best_state:
            raise ValueError("No trained model.")
        self.model.load_state_dict(self.best_state)
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device)).cpu().numpy().flatten()

    def plot_rmse_curve(self, best_epoch: int = None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_history["train_rmse"], label="Training RMSE", linewidth=2.5, color="#1f77b4")
        plt.plot(self.train_history["val_rmse"], label="Validation RMSE", linewidth=2.5, color="#ff7f0e")
        if best_epoch:
            plt.axvline(best_epoch - 1, color="red", linestyle="--", label=f"Best (ep {best_epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Book Sales RegressorMini — RMSE Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self, X_test, y_test, preprocessor):
        import matplotlib.pyplot as plt
        preds_scaled = self.predict(X_test)
        actual_scaled = y_test.cpu().numpy().flatten()
        pred_qty = preprocessor.inverse_transform_target(preds_scaled)
        actual_qty = preprocessor.inverse_transform_target(actual_scaled)

        coeffs = np.polyfit(pred_qty, actual_qty, 1)
        x_range = np.linspace(pred_qty.min(), pred_qty.max(), 100)

        plt.figure(figsize=(12, 8))
        plt.scatter(pred_qty, actual_qty, alpha=0.6, s=30, c='#1f77b4', label='Test samples')
        plt.plot(x_range, np.poly1d(coeffs)(x_range), 'r--', linewidth=3,
                 label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
        plt.plot(x_range, x_range, 'g-', alpha=0.8, label='Perfect')
        plt.xlabel('Predicted Quantity')
        plt.ylabel('Actual Quantity')
        plt.title('Test Set: Actual vs Predicted (Original Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# === Utility functions ===
def setup_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_random_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_split_data(data_path: Path, config: TrainingConfig):
    df_full = pd.read_csv(data_path, dtype={"isbn": "string", "q_since_first": "string"})
    df_full["isbn"] = df_full["isbn"].astype("string")
    df_train, df_temp = train_test_split(df_full, test_size=0.2, random_state=config.seed, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=config.seed, shuffle=True)
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    return df_train, df_val, df_test

def prepare_tensors(df, cols, y_vals, device):
    X = torch.from_numpy(df[cols].to_numpy(np.float32))
    y = torch.from_numpy(y_vals.astype(np.float32))
    return X, y


# === LOAD TRAINED MODEL FOR INFERENCE ===
def load_trained_model(
        model_path: Path,
        input_dim: int,
        dropout: float = 0.3,
        device: torch.device = None
) -> RegressorMini:
    """
    Load the trained RegressorMini model from a saved state_dict.

    Parameters
    ----------
    model_path : Path
        Path to the saved .pth file (e.g., "results/regressor_best.pth")
    input_dim : int
        Number of input features (must match the one used during training)
    dropout : float
        Dropout rate used during training (must match exactly)
    device : torch.device, optional
        Device to load the model on (defaults to CPU, then CUDA/MPS if available)

    Returns
    -------
    model : RegressorMini
        Model with loaded best weights, ready for evaluation or prediction
    """
    if device is None:
        device = setup_device()

    # 1. Re-instantiate the model with the exact same architecture
    model = RegressorMini(input_dim=input_dim, dropout=dropout).to(device)

    # 2. Load the saved state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 3. Set to evaluation mode
    model.eval()

    print(f"Successfully loaded model weights from: {model_path}")
    print(f"   → Input dimension: {input_dim} | Dropout: {dropout} | Device: {device}")

    return model

# === Main pipeline ===
def main():
    config = TrainingConfig()
    device = setup_device()
    set_random_seeds(config.seed)
    print(f"Using device: {device}")

    results_folder = Path("results")
    results_folder.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/target_series_new_with_features2.csv")
    target_books_path = Path("data/target_books_new.csv")

    df_train, df_val, df_test = load_and_split_data(data_path, config)

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

    df_train, df_val, df_test = preprocessor.safe_create_dummies(df_train, df_val, df_test)
    df_train_sel, feature_cols = preprocessor.select_features(df_train)
    df_val_sel = df_val[feature_cols]
    df_test_sel = df_test[feature_cols]

    X_train, y_train = prepare_tensors(df_train_sel, feature_cols, y_train_scaled, device)
    X_val, y_val = prepare_tensors(df_val_sel, feature_cols, y_val_scaled, device)
    X_test, y_test = prepare_tensors(df_test_sel, feature_cols, y_test_scaled, device)

    trainer = ModelTrainer(config, device)
    best_val_rmse, best_epoch = trainer.train(X_train, y_train, X_val, y_val)

    torch.save(trainer.best_state, results_folder / "regressor_best.pth")
    test_rmse = trainer.evaluate(X_test, y_test)
    print(f"Test RMSE (scaled): {test_rmse:.4f}")

    trainer.plot_rmse_curve(best_epoch)
    trainer.plot_actual_vs_predicted(X_test, y_test, preprocessor)

    # Final predictions
    df_target = pd.read_csv(target_books_path, dtype={"isbn": "string", "q_since_first": "string"})
    ids = df_target[["isbn", "title"]].copy()
    df_target = preprocessor.impute_missing_values(df_target)
    df_target = preprocessor.transform_features(df_target)
    df_target_dummy = preprocessor.safe_transform_dummies(df_target)
    df_target_dummy = df_target_dummy.reindex(columns=feature_cols, fill_value=0.0)
    X_target = torch.from_numpy(df_target_dummy[feature_cols].to_numpy(np.float32))

    preds = np.clip(preprocessor.inverse_transform_target(trainer.predict(X_target)), 0, None).round().astype(int)
    ids["pred_quantity"] = preds
    print(ids.head(8))
    # ids.to_csv(results_folder / "final_predictions.csv", index=False)
    print("Final predictions saved!")


    # Add the predicted quantity to the target_df
    df_target["quantity"] = preds
    #

    backtest = pd.read_csv("data/backtest_v2.csv",
                           dtype={
                               "isbn": "string",
                               "q_since_first": "string",
                           })
    backtest["isbn"] = backtest["isbn"].astype("string")
    concat = pd.concat([df_target, backtest])



    q_since_first_mapping = {
        "9781338896459": [5, 6, 7, 8], # dog_man, hardcover, 9781338236576
        "9781338896398": [9, 10, 11, 12], # cat_kid, hardcover, 9789813387386
        "9781529097153": [9, 10, 11, 12], # andy_griffiths, paperback, 9781529088601
        "9781338347258": [15, 16, 17, 18], # captain_underpants, hardcover, 9781338271508
        "9789810950286": [41, 42, 43, 44], # captain_underpants, paperback, 9789810731540
        "9789814918015": [12, 13, 14, 15], # captain_underpants, paperback, 9789810731540
        "9781913484521": [5, 6, 7, 8], # guinness, paperback, 9781913484385
        "9781913484552": [5, 6, 7, 8] # guinness, hardcover, 9781913484385
    }
    q_series_mapping = {
        "9781338896459": "9781338236576",
        "9781338896398": "9789813387386",
        "9781529097153": "9781529088601",
        "9781338347258": "9781338271508",
        "9789810950286": "9789810731540",
        "9789814918015": "9789810731540",
        "9781913484521": "9781913484385",
        "9781913484552": "9781913484385"
    }


    quarters = ["q1", "q2", "q3", "q4"]
    ######################## dog_man ########################
    # "9781338896459": Q5-8, # dog_man, hardcover, 9781338236576: Q5-8
    plt.figure(figsize=(8, 5))

    dog_man_target = concat.loc[concat["isbn"] == "9781338896459", "quantity"]
    dog_man_backtest = concat.loc[concat["isbn"] == "9781338236576", "quantity"]
    plt.plot(quarters, dog_man_target.tolist(), marker='o', label="9781338896459 (target)")
    plt.plot(quarters, dog_man_backtest.tolist(), marker='s', label="9781338236576 (backtest)")
    plt.title("Dog Man on Market for over a year since marketed")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()
    ######################## cat_kid ########################
    # "9781338896398": [9, 10, 11, 12], # cat_kid, hardcover,
    plt.figure(figsize=(8, 5))
    cat_kid_target = concat.loc[concat["isbn"] == "9781338896398", "quantity"]
    cat_kid_backtest = concat.loc[concat["isbn"] == "9789813387386", "quantity"]
    plt.plot(quarters, cat_kid_target.tolist(), marker='o', label="9781338896398 (target) (Q9-12)")
    plt.plot(quarters, cat_kid_backtest.tolist(), marker='s', label="9789813387386 (backtest) (Q)")
    plt.title("Cat Kid on Market for over a year since marketed")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()

    ######################## andy_griffiths ########################
    # "9781529097153": [9, 10, 11, 12], # andy_griffiths, paperback, 9781529088601
    plt.figure(figsize=(8, 5))
    andy_griffiths_target = concat.loc[concat["isbn"] == "9781529097153", "quantity"]
    andy_griffiths_backtest = concat.loc[concat["isbn"] == "9781529088601", "quantity"]
    plt.plot(quarters, andy_griffiths_target.tolist(), marker='o', label="9781529097153 (target) (Q9-12)")
    plt.plot(quarters, andy_griffiths_backtest.tolist(), marker='s', label="9781529088601 (backtest) (Q)")
    plt.title("Andy Griffiths on Market for over 2 years since marketed")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()

    ######################## captain_underpants1 ########################
    # "9781338347258": [15, 16, 17, 18],  # captain_underpants, hardcover, 9781338271508
    plt.figure(figsize=(8, 5))
    captain_underpants_target1 = concat.loc[concat["isbn"] == "9781338347258", "quantity"]
    captain_underpants_backtest1 = concat.loc[concat["isbn"] == "9781338271508", "quantity"]
    plt.plot(quarters, captain_underpants_target1.tolist(), marker='o', label="9781338347258 (target)")
    plt.plot(quarters, captain_underpants_backtest1.tolist(), marker='s', label="9781338271508 (backtest)")
    plt.title("Captain Underpants on Market for 15-18 quarters since first sale")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()

    ######################## captain_underpants2 ########################
    # "9789810950286": [41, 42, 43, 44],  # captain_underpants, paperback, 9789810731540 Q28-31
    plt.figure(figsize=(8, 5))
    captain_underpants_target2 = concat.loc[
        (concat["isbn"] == "9789810950286") & ((concat["q_since_first"].isin(range(28, 32))))
        , "quantity"
    ]
    captain_underpants_backtest2 = concat.loc[concat["isbn"] == "9789810731540", "quantity"]
    plt.plot(quarters, captain_underpants_target2.tolist(), marker='o', label="9789810950286 (target)")
    plt.plot(quarters, captain_underpants_backtest2.tolist(), marker='s', label="9789810731540 (backtest)")
    plt.title("Captain Underpants on Market for almost 4 years since first sale")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()


    ######################## captain_underpants3 ########################
    # "9789814918015": [12, 13, 14, 15],  # captain_underpants, paperback, 9789810731540
    plt.figure(figsize=(8, 5))
    captain_underpants_target2 = concat.loc[
        (concat["isbn"] == "9789810950286") & ((concat["q_since_first"].isin(range(28, 32))))
        , "quantity"
    ]
    captain_underpants_backtest2 = concat.loc[concat["isbn"] == "9789810731540", "quantity"]
    plt.plot(quarters, captain_underpants_target2.tolist(), marker='o', label="9789810950286 (target)")
    plt.plot(quarters, captain_underpants_backtest2.tolist(), marker='s', label="9789810731540 (backtest)")
    plt.title("Captain Underpants on Market for almost 4 years since first sale")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()

    ######################## guinness1 ########################
    # "9781913484521": [5, 6, 7, 8],  # guinness, paperback, 9781913484385
    plt.figure(figsize=(8, 5))
    guinness_target1 = concat.loc[concat["isbn"] == "9781913484521", "quantity"]
    guinness_backtest1 = concat.loc[concat["isbn"] == "9781913484385", "quantity"]
    plt.plot(quarters, guinness_target1.tolist(), marker='o', label="9781913484521 (target) (Q9-12)")
    plt.plot(quarters, guinness_backtest1.tolist(), marker='s', label="9781913484385 (backtest) (Q)")
    plt.title("Guiness on Market for over 4 years since marketed")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()

    ######################## guinness2 ########################
    # "9781913484552": [5, 6, 7, 8]  # guinness, hardcover, 9781913484385
    plt.figure(figsize=(8, 5))
    guinness_target2 = concat.loc[concat["isbn"] == "9781913484552", "quantity"]
    guinness_backtest2 = concat.loc[concat["isbn"] == "9781913484385", "quantity"]
    plt.plot(quarters, guinness_target2.tolist(), marker='o', label="9781913484552 (target) (Q9-12)")
    plt.plot(quarters, guinness_backtest2.tolist(), marker='s', label="9781913484385 (backtest) (Q)")
    plt.title("Guiness on Market for over 4 years since marketed")
    plt.ylabel("Quantity")
    plt.xlabel("Time since marketed")
    plt.legend()
    plt.show()

    print("end")

    pass


if __name__ == "__main__":
    main()