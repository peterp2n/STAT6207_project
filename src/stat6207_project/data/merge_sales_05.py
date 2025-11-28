import polars as pl
import polars.selectors as cs
from pathlib import Path

if __name__ == "__main__":

    html_folder = Path("data")
    merge4_path = Path("data") / "merged4.csv"
    merge4 = (
        pl.scan_csv(
            merge4_path, schema_overrides={"isbn": pl.Utf8}
        )
        .collect()
    )

    merge5_path = Path("data") / "merged5_std_dummy.csv"
    merge5_std_dummy = (
        pl.scan_csv(
            merge5_path, schema_overrides={"isbn": pl.Utf8}
        )
        .collect()
    )

    sales_data = (
        pl.scan_csv(
            "expand_quarterly_sales_retail.csv",
            schema_overrides={
                "barcode2": pl.Utf8,
                "Quarter_num": pl.Enum(["1", "2", "3", "4"]),
            }
        )
        .collect()
    )

    merge5_all = (
        merge5_std_dummy
        .join(
            sales_data,
            left_on="isbn",
            right_on="barcode2",
            how="left"
        )
        .with_row_index()
    )

    # merge5_sales.write_csv(Path("data") / "merged5_all.csv", include_bom=True)


    pass