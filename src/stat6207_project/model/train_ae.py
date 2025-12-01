import torch
import numpy as np
from pathlib import Path
from autoencoder_trainer import AutoEncoderTrainer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load data
# -------------------------------
data_folder = Path("data")
data_folder.mkdir(parents=True, exist_ok=True)
X_train_full = torch.from_numpy(np.load(data_folder / 'X_train.npy')).float()
X_test       = torch.from_numpy(np.load(data_folder / 'X_test.npy')).float()

print(f"Full train set: {X_train_full.shape}")
print(f"Test set:       {X_test.shape}")

# -------------------------------
# 2. Create a validation set from training data (highly recommended!)
# -------------------------------
# We'll use 10% of training data as validation → prevents overfitting
X_train, X_val = train_test_split(
    X_train_full,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

X_train = torch.tensor(X_train)
X_val   = torch.tensor(X_val)

print(f"→ Training set:   {X_train.shape}")
print(f"→ Validation set: {X_val.shape}")

input_dim = X_train.shape[1]

# -------------------------------
# 3. Initialize and train
# -------------------------------
trainer = AutoEncoderTrainer(
    input_dim=input_dim,
    encoding_dim=32,
    lr=0.0001
)

trainer.train(
    train_data=X_train,
    val_data=X_val,           # Now we have validation!
    epochs=120,
    batch_size=32,
    print_every=10
)

# -------------------------------
# 4. Final evaluation on held-out test set
# -------------------------------
test_mse = trainer.evaluate(X_test)
print(f"\nFinal Test MSE: {test_mse:.6f}")

# -------------------------------
# 5. Plot with realistic, readable y-ticks (perfect for your loss range)
# -------------------------------

results_folder = Path("ae_results")
results_folder.mkdir(parents=True, exist_ok=True)

trainer.plot_losses(
    title="Autoencoder Reconstruction Loss (Children's Books Features)",
    y_scale='linear',
    y_lim=(0.0, 0.02),                                          # Upper limit = 0.02
    y_ticks=[0.000, 0.002, 0.004, 0.006, 0.008, 0.010,
             0.012, 0.014, 0.016, 0.018, 0.020],                # dense, readable ticks
    y_tick_labels=['0.000', '0.002', '0.004', '0.006', '0.008',
                   '0.010', '0.012', '0.014', '0.016', '0.018', '0.020'],
    save_path=results_folder / "autoencoder_loss.png",
    figsize=(12, 6)
)

# Save encoder weights (useful for feature extraction in sales prediction)
trainer.save_weights('encoder', results_folder / 'encoder_weights.pth')