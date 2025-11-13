import polars as pl
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

def download_images(isbns: list, urls: list) -> None:

    for isbn, url in zip(isbns, urls):

        if not url:
            print(f"Skipping {isbn} (no thumbnail)")
            image_folder = Path("data") / "images" / "missing" / f"{isbn}"
            image_folder = image_folder.mkdir(parents=True, exist_ok=True)
            continue

        print(f"Downloading {isbn}")
        try:
            # Open the URL and read the response
            with urlopen(url) as response:
                image_data = response.read()
        except HTTPError as e:
            print(f"{isbn}: {e}")

        image_folder = Path("data") / "images" / "success" / f"{isbn}"
        image_name = image_folder / f"{isbn}.jpg"

        # Save the image to a file
        image_folder.mkdir(parents=True, exist_ok=True)
        with open(image_name, 'wb') as writer:
            writer.write(image_data)

        print(f"Downloaded {isbn}")

if __name__ == "__main__":
    merged = pl.scan_csv(Path("data") / "merged.csv",
                         schema_overrides={"barcode": pl.Utf8,
                                            "isbn": pl.Utf8,
                                            "isbn_10": pl.Utf8,
                                            "isbn_13": pl.Utf8})

    isbns = merged.select("isbn").collect()["isbn"].to_list()[:2]
    urls = merged.select("thumbnail").collect()["thumbnail"].to_list()[:2]
    # Download images
    download_images(isbns, urls)

    pass





