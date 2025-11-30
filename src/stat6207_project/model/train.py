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
trainer = AutoEncoderTrainer(input_dim=input_dim, encoding_dim=16, lr=0.0005)  # Tune as needed
trainer.train(X_train, epochs=100, batch_size=32, val_data=X_test)

# Evaluate
test_loss = trainer.evaluate(X_test)
print(f"Test Reconstruction Loss: {test_loss:.4f}")

# Get embeddings for downstream sales prediction
train_embeddings = trainer.get_embeddings(X_train)
test_embeddings = trainer.get_embeddings(X_test)
# Now train a regressor: e.g., nn.Linear(16, 1) on train_embeddings -> y_train