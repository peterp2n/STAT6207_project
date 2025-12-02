#!/usr/bin/env python3
"""
text_to_embeddings.py
→ Saves: isbn, text0, text1, text2, ..., text383
Perfect symmetry with your image script (img0, img1, ...)
"""

import polars as pl
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == "__main__":
    print("Loading tokenizer & model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)\n")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Optional: move to GPU/MPS if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}\n")

    # Load descriptions
    merge_desc = (
        pl.read_csv(Path("data") / "merged.csv", schema_overrides={"isbn": pl.Utf8})
        .select(["isbn", "description"])
        .fill_null("")  # Treat null descriptions as empty string
    )
    # merge_desc.write_csv(Path("data") / "merge_desc.csv", include_bom=True)

    print(f"Found {len(merge_desc):,} books with descriptions\n")

    isbn_list = []
    embedding_arrays = []

    print("Starting text embedding extraction...\n")

    for row in merge_desc.iter_rows(named=True):
        isbn = row["isbn"]
        description = row["description"]

        if not description.strip():  # Skip empty or whitespace-only
            print(f"Skipping {isbn} → empty description")
            continue

        # Tokenize
        encoded = tokenizer(
            description,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(**encoded)

        # Mean pooling + L2 normalization
        sentence_emb = mean_pooling(output, encoded['attention_mask'])
        sentence_emb = F.normalize(sentence_emb, p=2, dim=1)  # (1, 384)

        # To numpy
        emb_np = sentence_emb.cpu().numpy()  # shape (1, 384)

        isbn_list.append(isbn)
        embedding_arrays.append(emb_np)

        print(f"Processed {isbn}: {emb_np.shape}")

    # ========================= Batch Save with text0, text1, ... =========================
    if isbn_list:
        print(f"\nSaving {len(isbn_list)} embeddings...")

        # Stack all embeddings → (N, 384)
        embeddings_stacked = np.vstack(embedding_arrays)

        # Create column names: text0, text1, ..., text383
        text_columns = [f"text{i}" for i in range(384)]

        # Build final Polars DataFrame
        final_df = (
            pl.DataFrame({"isbn": isbn_list})
            .with_columns(pl.from_numpy(embeddings_stacked, schema=text_columns))
            .select(["isbn"] + text_columns)  # enforce order
            .unique(subset=["isbn"], keep="first")
        )

        # Save
        output_path = Path("data") / "text_embeddings.csv"
        final_df.write_csv(output_path, include_bom=True)

        print("\n" + "="*70)
        print(f"SUCCESS! Text embeddings saved")
        print(f"→ {output_path.resolve()}")
        print(f"→ Shape: {final_df.shape} → columns: isbn, text0, text1, ..., text383")
        print(f"→ L2-normalized, ready for multimodal concat")
        print("="*70)
    else:
        print("No valid descriptions found. Nothing saved.")