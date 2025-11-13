import numpy as np
import polars as pl
from pathlib import Path
from PIL import Image
from urllib.request import urlopen
from urllib.error import HTTPError


def download_images(isbns: list, urls: list) -> None:
    for isbn, url in zip(isbns, urls):
        if not url:
            print(f"Skipping {isbn} (no thumbnail)")
            missing_folder = Path("data") / "images" / "missing" / f"{isbn}"
            missing_folder.mkdir(parents=True, exist_ok=True)
            continue
        print(f"Downloading {isbn}")
        try:
            with urlopen(url) as response:
                image_data = response.read()
        except HTTPError as e:
            print(f"{isbn}: {e}")
            continue
        image_folder = Path("data") / "images" / "success" / f"{isbn}"
        image_name = image_folder / f"{isbn}.jpg"
        image_folder.mkdir(parents=True, exist_ok=True)
        with open(image_name, "wb") as writer:
            writer.write(image_data)
        print(f"Downloaded {isbn}")

def save_embeddings(success_root: Path,
                               output_root: Path,
                               image_size: tuple[int, int],
                               mode: str,
                               normalize: bool):
    """
    Iterate subfolders under success_root; each folder name treated as ISBN.
    Load a .jpg image, resize, convert to array, optionally normalize to [0,1],
    and save compressed .npz as <isbn>.npz under output_root.
    Returns list of (isbn, shape, saved_path).
    """
    processed = []
    if not success_root.exists():
        print(f"Missing root: {success_root}")
        return processed
    output_root.mkdir(parents=True, exist_ok=True)

    for folder in success_root.iterdir():
        if not folder.is_dir():
            continue
        isbn = folder.name
        # Prefer file matching folder name; fallback to first .jpg
        candidates = []
        exact = folder / f"{isbn}.jpg"
        if exact.exists():
            candidates = [exact]
        else:
            candidates = sorted(folder.glob("*.jpg"))
        if not candidates:
            print(f"No .jpg in {folder}")
            continue
        img_path = candidates[0]
        try:
            with Image.open(img_path) as im:
                im = im.convert(mode)
                if image_size:
                    im = im.resize(image_size)
                arr = np.asarray(im, dtype=np.float32)
        except Exception as e:
            print(f"Failed {img_path}: {e}")
            continue
        if normalize:
            arr /= 255.0
        save_path = output_root / f"{isbn}.npz"
        np.savez_compressed(save_path, image=arr, isbn=isbn, size=image_size, mode=mode)
        processed.append((isbn, arr.shape, save_path))
        print(f"Saved {save_path} shape={arr.shape}")
    print(f"Total embeddings: {len(processed)}")
    return processed


if __name__ == "__main__":
    merged = pl.scan_csv(Path("data") / "merged.csv",
                         schema_overrides={"barcode": pl.Utf8,
                                           "isbn": pl.Utf8,
                                           "isbn_10": pl.Utf8,
                                           "isbn_13": pl.Utf8})
    isbns = merged.select("isbn").collect()["isbn"].to_list()
    urls = merged.select("thumbnail").collect()["thumbnail"].to_list()
    # # download_images(isbns, urls)

    save_embeddings(success_root=Path("data") / "images" / "success",
        output_root= Path("data") / "images" / "embeddings",
        image_size= (224, 224),
        mode= "RGB",
        normalize = True
    )



    pass