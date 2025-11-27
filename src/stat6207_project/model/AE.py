import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------- Device Setup --------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "mps":
    print("MPS (Apple Silicon GPU) is active!\n")

# -------------------------- Load & Split Data --------------------------
data_folder = Path("data")
df = pd.read_csv(data_folder / "merged5_std_dummy_drop.csv")

numeric_df = df.drop(columns=['isbn', 'title', 'publication_date'], errors='ignore')
X_train, X_val = train_test_split(
    numeric_df.values, test_size=0.2, random_state=42, shuffle=True
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   X_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)

# -------------------------- Autoencoder Model --------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=numeric_df.shape[1], encoding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# -------------------------- Training Loop --------------------------
epochs = 200
train_losses = []
val_losses = []

print(f"Training autoencoder: {len(X_train)} train | {len(X_val)} validation samples")
print("Training in progress...\n")

for epoch in range(1, epochs + 1):
    # Training
    model.train()
    train_loss = 0.0
    for x, target in train_loader:
        x, target = x.to(device), target.to(device)
        recon = model(x)
        loss = criterion(recon, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_mse = train_loss / len(X_train)
    train_losses.append(train_mse)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, target in val_loader:
            x, target = x.to(device), target.to(device)
            recon = model(x)
            val_loss += criterion(recon, target).item() * x.size(0)
    val_mse = val_loss / len(X_val)
    val_losses.append(val_mse)

    # Print every 20 epochs
    if epoch % 20 == 0 or epoch == epochs:
        print(f"Epoch {epoch:3d}/{epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

print("\nTraining complete!")

# -------------------------- Final Plot --------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, color='blue', linewidth=2, label='Training MSE')
plt.plot(range(1, epochs + 1), val_losses,   color='orange', linewidth=2, label='Validation MSE')

plt.title('Autoencoder Reconstruction Loss', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------- Save Latent Features --------------------------
model.eval()
with torch.no_grad():
    full_tensor = torch.tensor(numeric_df.values, dtype=torch.float32).to(device)
    latent_features = model.encoder(full_tensor).cpu().numpy()

result_df = pd.DataFrame(latent_features, columns=[f"latent_{i}" for i in range(32)])
result_df["best_sellers_rank"] = df["best_sellers_rank"].values

output_path = data_folder / "autoencoder_latent_features.csv"
result_df.to_csv(output_path, index=False)

print(f"\nLatent features (32-dim) saved to:")
print(f"→ {output_path.resolve()}")