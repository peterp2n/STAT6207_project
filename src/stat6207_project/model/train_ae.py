import torch
import numpy as np
import pandas as pd
from pathlib import Path
from autoencoder_trainer import AutoEncoderTrainer
from sklearn.model_selection import train_test_split
from torch.nn.functional import normalize

def get_cos_sim(
        target_tensor,
        encoded_tensor
):
    norm_target = normalize(target_tensor, p=2, dim=1)
    norm_encoded = normalize(encoded_tensor, p=2, dim=1)
    cos_sim = torch.sum(norm_target * norm_encoded, dim=1)
    return cos_sim

# -------------------------------
# 1. Load data using pandas
# -------------------------------
data_folder = Path("data")
data_folder.mkdir(parents=True, exist_ok=True)

# Define dtypes to match previous Polars schema_overrides and casting
dtype_dict = {
    "isbn": str,
    "number_of_reviews": "float32"
}

cols_use = [
        'print_length', 'item_weight', 'length', 'width',
        'height', 'channel',
        'Quarters_since_first', 'Previous_quarter_qty', 'Current_quarter_qty',
        'Book_Flag',
        'Avg_discount_cleaned', 'book_format_board_book',
        'book_format_cards', 'book_format_hardcover',
        'book_format_library_binding', 'book_format_paperback',
        'reading_age_adolescence or above', 'reading_age_baby',
        'reading_age_preadolescence', 'reading_age_preschool',
        'reading_age_toddler', 'Quarter_num_1', 'Quarter_num_2',
        'Quarter_num_3', 'Quarter_num_4'
]

train = pd.read_csv(
    data_folder / "train_all_cols_v3.csv",
    dtype=dtype_dict
)[cols_use]

test = pd.read_csv(
    data_folder / "test_all_cols_v3.csv",
    dtype=dtype_dict
)[cols_use]

# -------------------------------
# 2. Prepare feature matrices
# -------------------------------
feature_columns = [col for col in cols_use if col not in ("isbn", "Next_Q1_log1p")]

X_train_full = torch.from_numpy(train[feature_columns].values).float()
X_test       = torch.from_numpy(test[feature_columns].values).float()

print(f"Full train set: {X_train_full.shape}")
print(f"Test set:       {X_test.shape}")

# -------------------------------
# 3. Create validation split
# -------------------------------
X_train_np, X_val_np = train_test_split(
    X_train_full.numpy(),
    test_size=0.1,
    random_state=42,
    shuffle=True
)

X_train = torch.from_numpy(X_train_np).float()
X_val   = torch.from_numpy(X_val_np).float()

print(f"→ Training set:   {X_train.shape}")
print(f"→ Validation set: {X_val.shape}")

input_dim = X_train.shape[1]

# -------------------------------
# 4. Train autoencoder
# -------------------------------
trainer = AutoEncoderTrainer(
    input_dim=input_dim,
    encoding_dim=8,
    lr=1e-4
)

trainer.train(
    train_data=X_train,
    val_data=X_val,
    epochs=100,
    batch_size=32,
    print_every=10
)

# -------------------------------
# 5. Final evaluation
# -------------------------------
test_mse = trainer.evaluate(X_test)
print(f"\nFinal Test MSE: {test_mse:.6f}")

# -------------------------------
# 6. Dynamic loss plot
# -------------------------------
results_folder = Path("ae_results")
results_folder.mkdir(parents=True, exist_ok=True)

# Extract loss histories (assumes trainer stores them as lists or np.ndarray)
train_losses = np.array(trainer.train_losses)
val_losses   = np.array(trainer.val_losses)

# Combine both curves to determine global range
all_losses = np.concatenate([train_losses, val_losses])
min_loss   = all_losses.min()
max_loss   = all_losses.max()

# Add a small margin above the highest loss and ensure lower bound is >= 0
upper_bound = max_loss * 1.15
lower_bound = 0.0

# Generate 11 evenly spaced ticks (including 0 and the upper bound)
y_ticks = np.linspace(lower_bound, upper_bound, 11)

# Format tick labels nicely (4 decimal places, remove unnecessary zeros)
y_tick_labels = [f"{tick:.4f}".rstrip("0").rstrip(".") if tick != 0 else "0" for tick in y_ticks]

# If the range is very small, switch to scientific notation for clarity
if upper_bound < 1e-3:
    y_tick_labels = [f"{tick:.1e}" for tick in y_ticks]

trainer.plot_losses(
    title="Autoencoder Reconstruction Loss (Children's Books Features)",
    y_scale='linear',
    y_lim=(lower_bound, upper_bound),
    y_ticks=y_ticks.tolist(),
    y_tick_labels=y_tick_labels,
    save_path=results_folder / "autoencoder_loss.png",
    figsize=(12, 6)
)

print("Training and dynamic plotting completed successfully.")

target_path = data_folder / "train_all_cols_v3.csv"

target = pd.read_csv(target_path, dtype=dtype_dict)[cols_use]
target_tensor = torch.from_numpy(target.to_numpy()).float()

encoded_tensor = trainer.get_embeddings(target_tensor)

cos_sim = trainer.get_reconstruction_similarity(X_train_full)

trainer.save_weights(part="encoder", path=results_folder / "encoder_weights.pth")

print("end")