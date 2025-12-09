from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import matplotlib.pyplot as plt

# ==================== PATH & LOAD DATA ====================
data_folder = Path("data")
df = pd.read_csv(data_folder / "target_series_new_with_features2.csv")

# Drop year_quarter completely – we only use q_since_first as time index
df = df.drop(columns=['year_quarter'], errors='ignore')

# Sort by isbn and q_since_first (critical!)
df = df.sort_values(['isbn', 'q_since_first']).reset_index(drop=True)

# Fill missing static book features (per ISBN median)
numeric_cols = ['print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'price']
for col in numeric_cols:
    df[col] = df.groupby('isbn')[col].transform(lambda x: x.fillna(x.median() if x.notna().any() else 0))



# Label-encode categorical columns (format, channel, series) - NOT isbn
cat_cols = ['format', 'channel', 'series']
label_maps = {}
for col in cat_cols:
    df[col], uniques = pd.factorize(df[col])
    label_maps[col] = uniques
    print(f"{col}: {len(uniques)} unique values")

# Also factorize isbn for indexing purposes (but not as a categorical feature)
df['isbn'], isbn_uniques = pd.factorize(df['isbn'])
label_maps['isbn'] = isbn_uniques

# Group by isbn to build clean time series
grouped = df.groupby('isbn')
series_list = []
lengths = []

for isbn_id, group in grouped:
    group = group.sort_values('q_since_first')  # already sorted, but safe

    values = group['quantity'].values.astype(np.float32)
    # Known future covariates: q_num (1-4) and avg_discount_rate
    time_features = group[['q_num', 'avg_discount_rate']].values.astype(np.float32)

    # Static features (constant per book)
    static_cat = group[['format', 'channel', 'series']].iloc[0].values.astype(np.int64)  # Only actual categorical features (NOT isbn)
    static_real = group[numeric_cols].iloc[0].values.astype(np.float32)

    series_list.append({
        'isbn': isbn_id,  # Store isbn separately for tracking
        'values': values,
        'time_features': time_features,
        'static_cat': static_cat,
        'static_real': static_real
    })
    lengths.append(len(values))

print(f"\nFound {len(series_list)} unique books")
print(f"Longest history: {max(lengths)} quarters")

# ==================== PAD ALL SERIES ====================
max_len = max(lengths)

past_values = np.zeros((len(series_list), max_len), dtype=np.float32)
past_time_features = np.zeros((len(series_list), max_len, 2), dtype=np.float32)  # q_num + discount
past_observed_mask = np.zeros((len(series_list), max_len), dtype=np.float32)

static_cat_all = []
static_real_all = []

for i, s in enumerate(series_list):
    L = lengths[i]
    past_values[i, :L] = s['values']
    past_time_features[i, :L] = s['time_features']
    past_observed_mask[i, :L] = 1.0
    static_cat_all.append(s['static_cat'])
    static_real_all.append(s['static_real'])

# To tensors
past_values_t = torch.tensor(past_values).unsqueeze(-1)          # (N, T, 1)
past_time_feat_t = torch.tensor(past_time_features)              # (N, T, 2)
past_mask_t = torch.tensor(past_observed_mask).unsqueeze(-1)      # (N, T, 1)

static_cat_t = torch.tensor(np.stack(static_cat_all), dtype=torch.long)     # (N, 3) - format, channel, series
static_real_t = torch.tensor(np.stack(static_real_all), dtype=torch.float32) # (N, 7)

# ==================== MODEL CONFIG ====================
config = TimeSeriesTransformerConfig(
    prediction_length=4,
    context_length=16,
    input_size=1,
    lags_sequence=[1, 2, 3, 4, 8, 12],           # quarterly + yearly patterns

    num_time_features=2,                         # q_num + avg_discount_rate
    num_static_categorical_features=3,           # format, channel, series (NOT isbn - it's an identifier)
    cardinality=[
        len(label_maps['format']),
        len(label_maps['channel']),
        len(label_maps['series'])
    ],
    embedding_dimension=[4, 4, 4],

    num_static_real_features=7,                  # print_length, weight, size, rating, price

    distribution_output="student_t",
    loss="nll",
    scaling=True,
    d_model=64,
    encoder_layers=4,
    decoder_layers=4,
    dropout=0.1,
    num_parallel_samples=200
)

model = TimeSeriesTransformerForPrediction(config)

# ==================== FREEZE LAYERS ====================
print("\n" + "=" * 60)
print("FREEZING MODEL LAYERS")
print("=" * 60)

# First, freeze all parameters
for name, param in model.named_parameters():
    param.requires_grad = False

# Then, unfreeze only the output/distribution layers
unfrozen_layers = []
for name, param in model.named_parameters():
    if 'output' in name.lower() or 'distribution' in name.lower() or 'parameter' in name.lower():
        param.requires_grad = True
        unfrozen_layers.append(name)
        print(f"✓ Unfrozen: {name}")

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
frozen_params = total_params - trainable_params

print("\n" + "=" * 60)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
print("=" * 60)

# ==================== TRAINING ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
model.to(device)

past_values_t = past_values_t.to(device)
past_time_feat_t = past_time_feat_t.to(device)
past_mask_t = past_mask_t.to(device)
static_cat_t = static_cat_t.to(device)
static_real_t = static_real_t.to(device)

# Only optimize unfrozen parameters
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=6e-4)
model.train()

required_past = config.context_length + max(config.lags_sequence)
print(f"\nTraining started (required past: {required_past} quarters)...")
print("=" * 60)

# Track training history
train_history = {'loss': [], 'count': []}

for epoch in range(30):
    total_loss = 0.0
    count = 0

    for i in range(len(series_list)):
        L = lengths[i]
        if L <= required_past + config.prediction_length:
            continue

        # Use most recent history
        start = max(0, L - required_past)
        pv = past_values_t[i:i+1, start:L]
        pt = past_time_feat_t[i:i+1, start:L]
        pm = past_mask_t[i:i+1, start:L]

        fv = past_values_t[i:i+1, L:L+4]          # next 4 quarters (if exist)
        ft = past_time_feat_t[i:i+1, L:L+4]

        if fv.shape[1] < 4:
            continue

        try:
            outputs = model(
                past_values=pv,
                past_time_features=pt,
                past_observed_mask=pm,
                static_categorical_features=static_cat_t[i:i+1],
                static_real_features=static_real_t[i:i+1],
                future_values=fv,
                future_time_features=ft
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1
        except Exception as e:
            print(f"Warning: Error processing series {i}: {e}")
            continue

    if count > 0:
        avg_loss = total_loss / count
        train_history['loss'].append(avg_loss)
        train_history['count'].append(count)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d} | Avg Loss: {avg_loss:.6f} | Samples: {count}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

# ==================== PLOT TRAINING CURVE ====================
if train_history['loss']:
    plt.figure(figsize=(10, 6))
    plt.plot(train_history['loss'], label="Training Loss", linewidth=2.5, color="#1f77b4")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (NLL)", fontsize=12)
    plt.title("TimeSeriesTransformer Training Loss (Frozen Layers)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_curve_frozen.png", dpi=150)
    print("\n✓ Training curve saved to results/training_curve_frozen.png")
    plt.show()

# ==================== PREDICTION (next 4 quarters) ====================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)
model.eval()
predictions = {}

with torch.no_grad():
    for i in range(len(series_list)):
        try:
            isbn_encoded = series_list[i]['isbn']
            isbn_str = label_maps['isbn'][isbn_encoded]  # back to original ISBN

            # Full history as past
            pv = past_values_t[i:i+1, :lengths[i]]
            pt = past_time_feat_t[i:i+1, :lengths[i]]
            pm = past_mask_t[i:i+1, :lengths[i]]

            # Build future time features for next 4 quarters
            last_q_num = int(pt[0, -1, 0].item()) if lengths[i] > 0 else 1
            next_q_nums = [((last_q_num + j - 1) % 4) + 1 for j in range(1, 5)]
            mean_discount = df[df['isbn'] == isbn_str]['avg_discount_rate'].mean()
            if np.isnan(mean_discount):
                mean_discount = 0.0

            future_time = torch.tensor([[q, mean_discount] for q in next_q_nums],
                                     dtype=torch.float32).unsqueeze(0).to(device)

            out = model.generate(
                past_values=pv,
                past_time_features=pt,
                past_observed_mask=pm,
                static_categorical_features=static_cat_t[i:i+1],
                static_real_features=static_real_t[i:i+1],
                future_time_features=future_time,
            )

            pred = out.sequences.mean(dim=1).cpu().numpy().flatten().round(1)
            title = df[df['isbn'] == isbn_str]['title'].iloc[0]
            predictions[isbn_str] = (title, pred)
        except Exception as e:
            print(f"Warning: Prediction error for series {i}: {e}")
            continue

# ==================== SHOW RESULTS ====================
print("\n" + "=" * 60)
print("PREDICTED NEXT 4 QUARTERS")
print("=" * 60 + "\n")
for isbn, (title, pred) in list(predictions.items())[:50]:
    print(f"{isbn} | {title[:60]:60s} → {pred.tolist()}")

if len(predictions) > 50:
    print(f"\n... and {len(predictions)-50} more books")

print(f"\n{'=' * 60}")
print(f"Total predictions generated: {len(predictions)}")
print("=" * 60)

# Save predictions
Path("results").mkdir(exist_ok=True)
pred_df = pd.DataFrame([
    (isbn, title, *pred.tolist())
    for isbn, (title, pred) in predictions.items()
], columns=['isbn', 'title', 'pred_Q1', 'pred_Q2', 'pred_Q3', 'pred_Q4'])

pred_df.to_csv("results/predictions_frozen_model.csv", index=False)
print(f"\n✓ Predictions saved to results/predictions_frozen_model.csv")

# Save model
torch.save(model.state_dict(), "results/timeseries_model_frozen.pth")
print(f"✓ Model saved to results/timeseries_model_frozen.pth")
