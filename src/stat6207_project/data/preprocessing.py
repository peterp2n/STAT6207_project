import polars as pl
from pathlib import Path
from remove_char import DataCleaner


if __name__ == "__main__":
    data_folder = Path("data")
    raw_path = data_folder / "amazon.csv"
    clean_path = data_folder / "amazon_cleaned.csv"

    amazon = pl.scan_csv(data_folder / "amazon.csv",
                         schema_overrides={"barcode": pl.Utf8,
                                            "isbn_10": pl.Utf8,
                                            "isbn_13": pl.Utf8}).collect()

    api = pl.scan_csv(data_folder / "books_api.csv",
                      schema_overrides={"barcode": pl.Utf8,
                                        "isbn_10": pl.Utf8,
                                        "isbn_13": pl.Utf8}).collect()

    if clean_path.exists():
        Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
        amazon.write_csv(clean_path, include_header=True)

    pass