from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from sqlalchemy import create_engine
import polars as pl

@dataclass
class Data:

    args : dict = field(default_factory=dict)
    table_holder: dict[str, pl.DataFrame] = field(default_factory=dict, init=False)
    _db_path: Optional[Path] = field(default=None, init=False)
    queries: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._db_path = self.args.get("db_path", "")

    def load_all_tables(self) -> None:
        """Load all tables from the database using SQLAlchemy."""

        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found: {self._db_path.resolve()}")

        engine = create_engine(f"sqlite:///{self._db_path}")
        table_names = {"products", "purchase", "sales", "shops"}

        with engine.connect() as conn:
            for table_name in table_names:
                query = f"SELECT * FROM {table_name}"
                try:
                    df = pl.read_database(query=query, connection=conn)
                except:
                    raise ValueError(f"Failed to load table: {table_name}")
                print(f"Loaded {len(df)} rows from table '{table_name}'")

                self.table_holder[table_name] = df.lazy()

    def load_queries(self):
        if all((self.table_holder, self.table_holder.get("products") is not None)):
            self.queries = ["9780007371082", "9780064450836"]
            # self.queries = (
            #     self.table_holder.get("products")
            #     .filter(
            #         (pl.col("barcode2").str.contains(r"^978")) & (pl.col("barcode2").str.len_chars() == 13)
            #     )
            #     .unique(maintain_order=True)
            #     .collect()["barcode2"]
            # )