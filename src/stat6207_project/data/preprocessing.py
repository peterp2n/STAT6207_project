import polars as pl
from pathlib import Path
from remove_char import DataCleaner


if __name__ == "__main__":
    data_folder = Path("data")
    raw_path = data_folder / "books_api.csv"
    clean_path = data_folder / "books_api_cleaned.csv"

    amazon = pl.scan_csv(data_folder / "amazon.csv",
                         schema_overrides={"barcode": pl.Utf8,
                                            "isbn_10": pl.Utf8,
                                            "isbn_13": pl.Utf8}).collect()

    api_clean = pl.scan_csv(data_folder / "books_api.csv",
                      schema_overrides={"barcode": pl.Utf8,
                                        "isbn": pl.Utf8,
                                        "isbn_13": pl.Utf8}).collect()

    cleaner = DataCleaner()
    amazon_clean = cleaner.clean_and_save(
        df=amazon,
        output_path=data_folder / 'amazon_cleaned.csv',
        deduplicate_on=['isbn_13']
    )
    # api_clean = cleaner.clean_and_save(
    #     df=api,
    #     output_path=data_folder / 'books_api_cleaned.csv',
    #     deduplicate_on=['isbn']
    # )

    Path(data_folder / "amazon_cleaned.csv").parent.mkdir(parents=True, exist_ok=True)
    Path(data_folder / "api_cleaned.csv").parent.mkdir(parents=True, exist_ok=True)
    amazon_clean.write_csv(data_folder / "amazon_cleaned.csv", include_header=True)
    api_clean.write_csv(data_folder / "books_api_cleaned.csv", include_header=True)

    merged = amazon.join(api_clean, how="outer", left_on="isbn_10", right_on="isbn")
    merged.write_csv(data_folder / "merged.csv", include_header=True)


    pass