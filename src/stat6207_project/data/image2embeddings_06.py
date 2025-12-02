#!/usr/bin/env python3
"""
image_to_embeddings.py
→ Saves CSV with columns:  isbn, img0, img1, img2, ..., img2047
Perfect match for your multimodal concat model!
"""

from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
from torchvision import models, transforms
from PIL import Image


# ========================== Configuration ==========================
INPUT_ROOT = Path("data") / "images" / "success"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ========================== Device ==========================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}\n")


# ========================== Model ==========================
print("Loading ResNet50 (ImageNet)...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
model.eval()

if device.type != "cpu":
    model = model.half()
model = model.to(device)


# ========================== Preprocess ==========================
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_embedding(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    if device.type != "cpu":
        tensor = tensor.half()

    with torch.no_grad():
        emb = model(tensor)                    # (1, 2048, 1, 1)
        emb = torch.flatten(emb, 1)            # (1, 2048)
        emb = F.normalize(emb, p=2, dim=1)     # L2 normalize
    return emb.cpu().numpy()                   # (1, 2048)


# ========================== Main ==========================
def main() -> None:
    if not INPUT_ROOT.exists() or not any(INPUT_ROOT.iterdir()):
        print(f"Error: '{INPUT_ROOT}' is empty or does not exist.")
        return

    isbn_list = []
    embedding_list = []

    print("Extracting image embeddings...\n")
    for isbn_folder in sorted(INPUT_ROOT.iterdir()):
        if not isbn_folder.is_dir():
            continue
        isbn = isbn_folder.name
        if not (isbn.isdigit() and len(isbn) == 13):
            continue

        img_path = isbn_folder / f"{isbn}.jpg"
        if not img_path.is_file():
            print(f"Missing: {img_path}")
            continue

        try:
            print(f"Processing {isbn}")
            emb = get_embedding(img_path)           # (1, 2048)
            isbn_list.append(isbn)
            embedding_list.append(emb)
        except Exception as e:
            print(f"Failed {isbn}: {e}")

    if not isbn_list:
        print("No images processed.")
        return

    # Stack embeddings → (N, 2048)
    embeddings_np = np.vstack(embedding_list)           # shape (N, 2048)

    # Create column names: img0, img1, ..., img2047
    img_columns = [f"img{i}" for i in range(2048)]

    # Build final Polars DataFrame
    final_df = (
        pl.DataFrame({"isbn": isbn_list})
        .with_columns(pl.from_numpy(embeddings_np, schema=img_columns))
        .unique(subset=["isbn"], keep="first")
    )

    # Ensure isbn is first, then img0..img2047
    final_df = final_df.select(["isbn"] + img_columns)

    # Save
    output_path = OUTPUT_DIR / "images_embeddings.csv"
    final_df.write_csv(output_path, include_bom=True)

    print("\n" + "="*70)
    print(f"Done! {len(isbn_list)} books processed")
    print(f"Saved → {output_path.resolve()}")
    print(f"Shape : {final_df.shape}  →  columns: isbn, img0, img1, ..., img2047")
    print("="*70)


if __name__ == "__main__":
    main()