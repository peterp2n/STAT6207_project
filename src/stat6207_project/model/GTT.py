from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# ==================== PATH & LOAD DATA ====================
data_folder = Path("data")
df = pd.read_csv(data_folder / "target_series_new_with_features2.csv")


target_col = 'quantity'

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

# ==================== TIME SERIES SPLIT ====================
print("\n" + "=" * 60)
print("TIME SERIES SPLIT SETUP")
print("=" * 60)

# Use TimeSeriesSplit to create train/validation splits
# We'll use the last split for final training
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# Create indices array for splitting
series_indices = np.arange(len(series_list))

print(f"Using {n_splits}-fold Time Series Split")
print(f"Total series: {len(series_list)}")

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

required_past = config.context_length + max(config.lags_sequence)

# Track history across all folds
all_fold_history = []

# Use the last split for final training
for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(series_indices), 1):
    print("\n" + "=" * 60)
    print(f"TIME SERIES FOLD {fold_idx}/{n_splits}")
    print("=" * 60)
    print(f"Train series: {len(train_idx)} | Validation series: {len(val_idx)}")

    # Convert to sets for faster lookup
    train_set = set(train_idx)
    val_set = set(val_idx)

    model.train()
    train_history = {'loss': [], 'val_loss': [], 'count': [], 'val_count': []}

    print(f"Training started (required past: {required_past} quarters)...")

    for epoch in range(30):
        # Training phase
        total_train_loss = 0.0
        train_count = 0

        for i in train_idx:
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

                total_train_loss += loss.item()
                train_count += 1
            except Exception as e:
                if epoch == 0:  # Only print first epoch
                    print(f"Warning: Error processing train series {i}: {e}")
                continue

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for i in val_idx:
                L = lengths[i]
                if L <= required_past + config.prediction_length:
                    continue

                start = max(0, L - required_past)
                pv = past_values_t[i:i+1, start:L]
                pt = past_time_feat_t[i:i+1, start:L]
                pm = past_mask_t[i:i+1, start:L]

                fv = past_values_t[i:i+1, L:L+4]
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

                    total_val_loss += outputs.loss.item()
                    val_count += 1
                except Exception as e:
                    if epoch == 0:
                        print(f"Warning: Error processing val series {i}: {e}")
                    continue

        model.train()

        # Record losses
        if train_count > 0:
            avg_train_loss = total_train_loss / train_count
            train_history['loss'].append(avg_train_loss)
            train_history['count'].append(train_count)

        if val_count > 0:
            avg_val_loss = total_val_loss / val_count
            train_history['val_loss'].append(avg_val_loss)
            train_history['val_count'].append(val_count)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_str = f"Train: {avg_train_loss:.6f} ({train_count})" if train_count > 0 else "Train: N/A"
            val_str = f"Val: {avg_val_loss:.6f} ({val_count})" if val_count > 0 else "Val: N/A"
            print(f"Epoch {epoch+1:2d} | {train_str} | {val_str}")

    all_fold_history.append({
        'fold': fold_idx,
        'train_history': train_history,
        'train_idx': train_idx,
        'val_idx': val_idx
    })

    print(f"\nFold {fold_idx} complete!")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

# Print fold summaries
print("\n" + "=" * 60)
print("FOLD SUMMARIES")
print("=" * 60)
for fold_data in all_fold_history:
    fold = fold_data['fold']
    history = fold_data['train_history']
    if history['loss'] and history['val_loss']:
        final_train = history['loss'][-1]
        final_val = history['val_loss'][-1]
        print(f"Fold {fold}: Final Train Loss = {final_train:.6f} | Final Val Loss = {final_val:.6f}")

# ==================== PLOT TRAINING CURVE ====================
print("\n" + "=" * 60)
print("Plotting Training Curves")
print("=" * 60)

fig, axes = plt.subplots(1, n_splits, figsize=(6*n_splits, 5))
if n_splits == 1:
    axes = [axes]

for fold_data, ax in zip(all_fold_history, axes):
    fold = fold_data['fold']
    history = fold_data['train_history']

    if history['loss']:
        ax.plot(history['loss'], label="Training Loss", linewidth=2.5, color="#1f77b4")
    if history['val_loss']:
        ax.plot(history['val_loss'], label="Validation Loss", linewidth=2.5, color="#ff7f0e")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (NLL)", fontsize=11)
    ax.set_title(f"Fold {fold}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_curves_tscv_frozen.png", dpi=150)
print("✓ Training curves saved to results/training_curves_tscv_frozen.png")
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

# Now my target_df is called
target_df = pd.read_csv(data_folder / "target_books_new.csv")
# My target_df contains 8 isbns with different values of q_since_first, as well as other static features.
# I want to generate predictions for these books using the trained model.
# create a new column called 'predicted_quantity' in target_df
# Just use the trained model to generate predictions for these books
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS FOR TARGET BOOKS")
print("=" * 60)
model.eval()
target_predictions = []
with torch.no_grad():
    for idx, row in target_df.iterrows():
        try:
            isbn_str = row['isbn']
            if isbn_str not in label_maps['isbn']:
                print(f"Warning: ISBN {isbn_str} not in training data.")
                continue
            isbn_encoded = np.where(label_maps['isbn'] == isbn_str)[0][0]

            # Extract series data
            series_data = next((s for s in series_list if s['isbn'] == isbn_encoded), None)
            if series_data is None:
                print(f"Warning: No series data for ISBN {isbn_str}.")
                continue

            L = len(series_data['values'])

            # Full history as past
            pv = past_values_t[isbn_encoded:isbn_encoded+1, :L]
            pt = past_time_feat_t[isbn_encoded:isbn_encoded+1, :L]
            pm = past_mask_t[isbn_encoded:isbn_encoded+1, :L]

            # Build future time features for next 4 quarters
            last_q_num = int(pt[0, -1, 0].item()) if L > 0 else 1
            next_q_nums = [((last_q_num + j - 1) % 4) + 1 for j in range(1, 5)]
            mean_discount = row['avg_discount_rate']
            if np.isnan(mean_discount):
                mean_discount = 0.0

            future_time = torch.tensor([[q, mean_discount] for q in next_q_nums],
                                     dtype=torch.float32).unsqueeze(0).to(device)

            out = model.generate(
                past_values=pv,
                past_time_features=pt,
                past_observed_mask=pm,
                static_categorical_features=static_cat_t[isbn_encoded:isbn_encoded+1],
                static_real_features=static_real_t[isbn_encoded:isbn_encoded+1],
                future_time_features=future_time,
            )

            pred = out.sequences.mean(dim=1).cpu().numpy().flatten().round(1)
            target_predictions.append((isbn_str, row['title'], pred))
        except Exception as e:
            print(f"Warning: Prediction error for ISBN {isbn_str}: {e}")
            continue

print("end")