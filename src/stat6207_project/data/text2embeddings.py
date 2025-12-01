import polars as pl
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == "__main__":

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    merge_desc = (
        pl.read_csv(Path("data") / "merged.csv", schema_overrides={"isbn": pl.Utf8})
        .select(["isbn", "description"])
    )
    # merge_desc.write_csv(Path("data") / "merge_desc.csv", include_bom=True)


    embeddings_path = Path("data") / "text" / "embeddings"
    embeddings_path.mkdir(parents=True, exist_ok=True)

    # Containers for batch collection
    isbn_list = []          # Will become (n, 1) DataFrame
    embedding_arrays = []   # List of (1, 384) numpy arrays → later vstack

    for row in merge_desc.iter_rows(named=True):
        isbn = row.get("isbn")
        description = row.get("description", "")

        # Skip if description or isbn is null/empty
        if not description or not isbn:
            print(f"ISBN {isbn}: Skipping (description is null/empty)")
            continue

        # Tokenize sentences
        encoded_input = tokenizer(description, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Convert to numpy (shape: (1, 384))
        embedding_np = sentence_embeddings.cpu().numpy()

        # Collect for later batch save
        isbn_list.append(isbn)
        embedding_arrays.append(embedding_np)

        print(f"Processed ISBN {isbn}: shape {embedding_np.shape}")

    # ------------------------------------------------------------------
    # Batch saving logic (replaces the previous per-row saving)
    # ------------------------------------------------------------------
    if isbn_list:  # Only proceed if we have at least one valid row
        # 1. ISBN column as (n, 1) Polars DataFrame
        isbn_df = pl.DataFrame({"isbn": isbn_list})

        # 2. Vertically stack all embeddings → (n, 384) numpy array
        embeddings_stacked = np.vstack(embedding_arrays)  # shape: (n, 384)

        # Convert stacked embeddings to Polars DataFrame with generic column names
        embedding_cols = [f"dim_{i}" for i in range(embeddings_stacked.shape[1])]
        embeddings_df = pl.DataFrame(embeddings_stacked, schema=embedding_cols)

        # 3. Horizontally concatenate isbn with embeddings
        final_df = isbn_df.hstack(embeddings_df)

        # 4. Save single CSV with BOM
        output_csv = embeddings_path / "text_embeddings.csv"
        final_df.write_csv(output_csv, include_bom=True)

        print(f"\nAll embeddings saved to {output_csv}")
        print(f"Final shape: {final_df.shape} (rows, columns)")
    else:
        print("No valid descriptions found. Nothing was saved.")