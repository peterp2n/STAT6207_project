# train_ae2.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from autoencoder_trainer import AutoEncoderTrainer
import matplotlib.pyplot as plt

# ================================
# 1. Set up paths and device
# ================================
data_folder = Path("data")
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

# Auto-detect device: MPS if available (Apple Silicon), else CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# ================================
# 2. Load and prepare data
# ================================
print("Loading data...")

# Define columns to use (adjust if needed)
cols_use = [
    'print_length', 'item_weight', 'length', 'width', 'height',
    'rating', 'number_of_reviews', 'price', 'Quarters_since_first',
    'Previous_quarter_qty', 'Current_quarter_qty', 'Avg_discount_cleaned',
    'book_format_board_book', 'book_format_cards', 'book_format_hardcover',
    'book_format_library_binding', 'book_format_paperback',
    'reading_age_adolescence or above', 'reading_age_baby',
    'reading_age_preadolescence', 'reading_age_preschool', 'reading_age_toddler',
    'Quarter_num_1', 'Quarter_num_2', 'Quarter_num_3', 'Quarter_num_4'
]

# Load full training data
train_path = data_folder / "train_all_cols_v3.csv"
df = pd.read_csv(train_path, usecols=cols_use)

print(f"Loaded {len(df):,} rows with {len(cols_use)} features")

# Handle missing values (simple forward/backward fill + zero)
df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

# Convert to numpy → torch tensor
X = df[cols_use].values.astype(np.float32)
X_tensor = torch.from_numpy(X)

# Train / Val / Test split (using indices to avoid data leakage across books)
train_idx, temp_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

X_train = X_tensor[train_idx]
X_val   = X_tensor[val_idx]
X_test  = X_tensor[test_idx]

# Move all splits to correct device
X_train = X_train.to(device)
X_val   = X_val.to(device)
X_test  = X_test.to(device)

print(f"Train set:      {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set:       {X_test.shape}")

input_dim = X_train.shape[1]

# ================================
# 3. Initialize and train autoencoder
# ================================
print("\nInitializing AutoEncoderTrainer...")

trainer = AutoEncoderTrainer(
    input_dim=input_dim,
    encoding_dim=8,        # You were using 8 before → keep it
    lr=1e-4
)

print(f"Starting training on {device}...")
trainer.train(
    train_data=X_train,
    val_data=X_val,
    epochs=100,
    batch_size=64,
    print_every=10
)

# ================================
# 4. Final evaluation
# ================================
test_mse = trainer.evaluate(X_test)
print(f"\nFinal Test MSE: {test_mse:.6f}")

# ================================
# 5. Plot training curves
# ================================
trainer.plot_losses(
    title="Book Features Autoencoder - Training & Validation Loss",
    save_path=results_folder / "autoencoder_loss_curve.png",
    y_scale='auto',
    y_lim="auto"
)

# ================================
# 6. Save encoder weights
# ================================
encoder_path = results_folder / "encoder_weights.pth"
trainer.save_weights(part="encoder", path=encoder_path)

# ================================
# 7. Extract embeddings from full training set
# ================================
print("Extracting embeddings from full training data...")
with torch.no_grad():
    full_embeddings = trainer.get_embeddings(X_tensor.to(device))  # Ensure full data is on device
    full_embeddings = full_embeddings.cpu().numpy()

# Save embeddings
embeddings_path = results_folder / "book_embeddings.npy"
np.save(embeddings_path, full_embeddings)
print(f"Embeddings saved to {embeddings_path} → shape: {full_embeddings.shape}")

# Optional: reconstruction similarity check
print("Computing reconstruction similarity on training set...")
cos_sim = trainer.get_reconstruction_similarity(X_train)
print(f"Mean cosine similarity (train): {cos_sim.mean():.4f} ± {cos_sim.std():.4f}")

print("\nTraining complete! Results saved in:", results_folder)