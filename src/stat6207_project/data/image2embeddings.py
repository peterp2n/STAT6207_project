#!/usr/bin/env python3
"""
image_to_embeddings.py
2025-ready, pathlib-only, zero deprecation warnings

Extracts L2-normalized 2048-dim ResNet50 embeddings from book covers
and saves them in the same format as your text pipeline:
→ ./image_embeddings/embeddings.csv (isbn, dim_0, ..., dim_2047)

Supports: MPS (Apple Silicon) > CUDA > CPU (automatic best device)
"""

from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
from torchvision import models, transforms
from PIL import Image


# ========================== Configuration ==========================
INPUT_ROOT = Path("data") / "images" / "success"                    # ISBN folders with .jpg files
OUTPUT_DIR = Path("data") / "images" / "embeddings"           # Final CSV will be saved here
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)     # pathlib version of os.makedirs


# ========================== Device Selection ==========================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()
print(f"Using device: {device}\n")


# ========================== Model (modern weights) ==========================
print("Loading ResNet50 with ImageNet weights...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Or use: weights=models.ResNet50_Weights.DEFAULT for the very latest

# Remove final classification layer → 2048-dim global average pooled features
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Use half precision on GPU/MPS for speed & lower memory
if device.type != "cpu":
    model = model.half()

model = model.to(device)


# ====================== Preprocessing (no deprecated transforms) ======================
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),                      # PIL → torch.Tensor (uint8, CHW)
    transforms.ConvertImageDtype(torch.float32),   # uint8 → float32 and divide by 255
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def get_embedding(image_path: Path) -> np.ndarray:
    """
    Returns L2-normalized embedding as numpy array of shape (1, 2048).
    """
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    if device.type != "cpu":
        tensor = tensor.half()   # match model precision

    with torch.no_grad():
        emb = model(tensor)                    # (1, 2048, 1, 1)
        emb = torch.flatten(emb, 1)            # (1, 2048)
        emb = F.normalize(emb, p=2, dim=1)     # L2 normalize

    return emb.cpu().numpy()                   # (1, 2048) numpy array


# ========================= Main Processing =========================
def main() -> None:
    if not INPUT_ROOT.exists() or not any(INPUT_ROOT.iterdir()):
        print(f"Error: Directory '{INPUT_ROOT}' does not exist or is empty.")
        return

    isbn_list: list[str] = []
    embedding_arrays: list[np.ndarray] = []

    print("Starting image embedding extraction...\n")

    # Sorted for reproducible order
    for isbn_folder in sorted(INPUT_ROOT.iterdir()):
        if not isbn_folder.is_dir():
            continue

        isbn = isbn_folder.name

        # Validate ISBN-13
        if not (isbn.isdigit() and len(isbn) == 13):
            print(f"Skipping non-ISBN folder: {isbn}")
            continue

        image_path = isbn_folder / f"{isbn}.jpg"
        if not image_path.is_file():
            print(f"Missing image: {image_path}")
            continue

        try:
            print(f"Processing {isbn} ...")
            emb_np = get_embedding(image_path)
            isbn_list.append(isbn)
            embedding_arrays.append(emb_np)
        except Exception as e:
            print(f"Failed {isbn}: {e}")

    # ========================= Batch Save (identical to text pipeline) =========================
    if isbn_list:
        # ISBN column
        isbn_df = pl.DataFrame({"isbn": isbn_list})

        # Stack all embeddings → (n, 2048)
        embeddings_np = np.vstack(embedding_arrays)
        dim_columns = [f"dim_{i}" for i in range(2048)]
        embeddings_df = pl.DataFrame(embeddings_np, schema=dim_columns)

        # Combine and save
        final_df = isbn_df.hstack(embeddings_df)
        output_csv = OUTPUT_DIR / "images_embeddings.csv"
        final_df.write_csv(output_csv, include_bom=True)

        print("\n" + "="*60)
        print(f"Success! Processed {len(isbn_list)} images")
        print(f"L2-normalized ResNet50 embeddings saved to:")
        print(f"   {output_csv.resolve()}")
        print(f"Shape: {final_df.shape} (rows × columns)")
    else:
        print("No images were successfully processed.")


if __name__ == "__main__":
    main()