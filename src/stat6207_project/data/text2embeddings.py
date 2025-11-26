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

    merge2_desc = pl.read_csv(Path("data") / "merged2_desc.csv").with_columns(pl.col("isbn").cast(pl.Utf8))

    embeddings_path = Path("data") / "text" / "embeddings"
    embeddings_path.mkdir(parents=True, exist_ok=True)

    for row in merge2_desc.iter_rows(named=True):
        isbn = row.get("isbn")
        description = row.get("description", "")
        title = row.get("title", "")

        # Skip if description is null or empty
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

        # Convert to numpy and save
        embedding_np = sentence_embeddings.cpu().numpy()  # Shape: (1, 384)

        # Save with embeddings_ prefix
        save_path = embeddings_path / f"embeddings_{isbn}.npy"
        np.save(save_path, embedding_np)

        print(f"Saved embedding for ISBN {isbn}: shape {embedding_np.shape}")
