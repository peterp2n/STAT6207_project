import numpy as np
import polars as pl
from pathlib import Path
from remove_char import DataCleaner
from src.stat6207_project.scraper.Data import Data


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

    arguments = {
        "db_path": Path("data") / "Topic1_dataset.sqlite",
        "headless": False,  # Set to True for headless mode
        "json_folder": Path("data") / "scrapes",
    }
    data = Data(arguments)
    data.load_all_tables()
    products = data.table_holder.get("products").collect()
    data_isbns = products["barcode2"].to_list()

    amazon_isbns = amazon_clean["isbn_13"].to_list()
    api_isbns = api_clean["isbn"].to_list()

    data_amazon_common = np.isin(data_isbns, amazon_isbns)
    data_api_common = np.isin(data_isbns, api_isbns)

    data_amazon_count, data_amazon_rate = data_amazon_common.sum(), data_amazon_common.mean()
    data_api_count, data_api_rate = data_api_common.sum(), data_api_common.mean()

    # api_clean = cleaner.clean_and_save(
    #     df=api,
    #     output_path=data_folder / 'books_api_cleaned.csv',
    #     deduplicate_on=['isbn']
    # )

    Path(data_folder / "amazon_cleaned.csv").parent.mkdir(parents=True, exist_ok=True)
    Path(data_folder / "api_cleaned.csv").parent.mkdir(parents=True, exist_ok=True)
    # amazon_clean.write_csv(data_folder / "amazon_cleaned.csv", include_header=True)
    # api_clean.write_csv(data_folder / "books_api_cleaned.csv", include_header=True)

    merged = api_clean.join(amazon, how="left", left_on="isbn", right_on="isbn_10")
    merged.write_csv(data_folder / "merged.csv", include_header=True)


    pass