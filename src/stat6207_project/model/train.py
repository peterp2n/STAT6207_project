import torch
import numpy as np
from pathlib import Path
from autoencoder_trainer import AutoEncoderTrainer

# Load pre-saved data (assuming X is features like concatenated book metadata + past sales)
data_folder = Path("data")
X_train = torch.from_numpy(np.load(data_folder / 'X_train.npy')).float()
X_test = torch.from_numpy(np.load(data_folder / 'X_test.npy')).float()
# y_train/y_test would be used later for a regressor, e.g., predicting Next_Q1 sales

# Determine input_dim from data (e.g., book features: print_length, rating, price, etc.)
input_dim = X_train.shape[1]

# Initialize and train
trainer = AutoEncoderTrainer(input_dim=input_dim, encoding_dim=32, lr=0.0005)  # Tune as needed

trainer.train(
    train_data=X_train,
    val_data=X_test,           # highly recommended!
    epochs=200,
    batch_size=64,
    print_every=20
)

# Final test performance
test_mse = trainer.evaluate(X_test)
print(f"Final Test MSE: {test_mse:.6f}")

trainer.plot_losses(
    title="Autoencoder MSE (Log Scale) - Children's Books Features",
    y_scale='log',
    y_lim=(1e-6, 1.0),
    y_ticks=[1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0],
    save_path="ae_loss_log_scale.png"
)