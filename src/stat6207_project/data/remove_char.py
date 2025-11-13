import polars as pl
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Callable


class DataCleaner:
    INVISIBLE_CHARS = {
        '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
        '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
        '\ufeff', '\u00a0'
    }
    CONVERSION_FACTORS = {
        'cm_to_inches': 0.393701,
        'pounds_to_grams': 453.592,
        'ounces_to_grams': 28.3495
    }
    PATTERNS = {
        'dimensions': r'([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)\s*(inches?|cm|centimeter|centimeters)?',
        'rating': r'([\d.]+)\s+out of',
        'rating_base': r'([\d.]+)\s+out of 5',
        'number': r'([\d,]+)',
        'price': r'([\d,]+\.?\d*)',
        'age_range': r'(\d+)\s*-\s*(\d+)',
        'pounds': r'([\d.]+)\s*pounds?',
        'ounces': r'([\d.]+)\s*ounces?',
    }
    ID_COLUMNS = ['isbn_10', 'isbn_13', 'barcode', 'asin']

    def __init__(self):
        self.clean_map: Dict[str, Callable] = {
            'asin': self.clean_text,
            'author': self.clean_text,
            'availability': self.clean_text,
            'best_sellers_rank': self.clean_text,
            'book_format': self.clean_text,
            'description': self.clean_text,
            'edition': self.clean_text,
            'features': self.clean_text,
            'language': self.clean_text,
            'part_of_series': self.clean_text,
            'product_name': self.clean_text,
            'product_url': self.clean_text,
            'publisher': self.clean_text,
            'isbn_10': self.clean_isbn,
            'isbn_13': self.clean_isbn,
            'barcode': self.clean_isbn,
            'number_of_reviews': self.clean_numeric,
            'print_length': self.clean_numeric,
            'rating': self.clean_rating,
            'customer_reviews': self.clean_base_rating,
            'publication_date': self.clean_publication_date,
            'item_weight': self.clean_item_weight,
            'price': self.clean_price,
            'reading_age': self.clean_reading_age,
        }

    @staticmethod
    def is_valid_text(x: Optional[str]) -> bool:
        """Check if text is valid (not None and not empty after stripping)."""
        return x is not None and isinstance(x, str) and bool(x.strip())

    def clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean text by removing invisible characters."""
        if not self.is_valid_text(text):
            return None
        cleaned = self.remove_invisible_chars(text)
        return cleaned if cleaned else None

    def clean_isbn(self, text: Optional[str]) -> Optional[str]:
        """Clean ISBN by removing invisible characters and non-digits."""
        if not self.is_valid_text(text):
            return None
        text = self.remove_invisible_chars(text)
        text = text.replace('.0', '')
        cleaned = re.sub(r'\D', '', text)
        return cleaned if cleaned else None

    def clean_numeric(self, text: Optional[str]) -> Optional[int]:
        """Extract numeric value from text."""
        if not self.is_valid_text(text):
            return None
        match = re.search(self.PATTERNS['number'], text)
        if match:
            try:
                return int(match.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    def clean_rating(self, text: Optional[str]) -> Optional[float]:
        """Extract rating from text (e.g., '4.5 out of')."""
        if not self.is_valid_text(text):
            return None
        match = re.search(self.PATTERNS['rating'], text)
        if match:
            try:
                return round(float(match.group(1)), 1)
            except ValueError:
                return None
        return None

    def clean_base_rating(self, text: Optional[str]) -> Optional[float]:
        """Extract rating from text (e.g., '4.5 out of 5')."""
        if not self.is_valid_text(text):
            return None
        match = re.search(self.PATTERNS['rating_base'], text)
        if match:
            try:
                return round(float(match.group(1)), 1)
            except ValueError:
                return None
        return None

    def clean_publication_date(self, text: Optional[str]) -> Optional[str]:
        """Parse publication date and convert to ISO format."""
        if not self.is_valid_text(text):
            return None
        try:
            dt = datetime.strptime(text, '%B %d, %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return None

    def clean_item_weight(self, text: Optional[str]) -> Optional[float]:
        """Convert item weight to grams."""
        if not self.is_valid_text(text):
            return None
        total_grams = 0.0
        try:
            p_match = re.search(self.PATTERNS['pounds'], text)
            o_match = re.search(self.PATTERNS['ounces'], text)
            if p_match:
                total_grams += float(p_match.group(1)) * self.CONVERSION_FACTORS['pounds_to_grams']
            if o_match:
                total_grams += float(o_match.group(1)) * self.CONVERSION_FACTORS['ounces_to_grams']
            return round(total_grams, 2) if total_grams else None
        except Exception:
            return None

    def clean_price(self, text: Optional[str]) -> Optional[float]:
        """Extract price from text."""
        if not self.is_valid_text(text):
            return None
        match = re.search(self.PATTERNS['price'], text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    def clean_reading_age(self, text: Optional[str]) -> Optional[str]:
        """Extract reading age range from text."""
        if not self.is_valid_text(text):
            return None
        match = re.search(self.PATTERNS['age_range'], text)
        return f"{match.group(1)} - {match.group(2)}" if match else None

    @staticmethod
    def remove_invisible_chars(text: str) -> str:
        """Remove invisible and control characters from text."""
        return ''.join(
            c for c in text
            if c not in DataCleaner.INVISIBLE_CHARS and unicodedata.category(c)[0] != 'C'
        ).strip()

    @classmethod
    def parse_dimensions(cls, dimension_str: Optional[str]) -> Dict[str, Optional[float]]:
        """Parse dimension string and convert to inches."""
        result = {'length': None, 'width': None, 'height': None}
        if not dimension_str or not isinstance(dimension_str, str):
            return result

        dimension_str = cls.remove_invisible_chars(dimension_str)
        pattern = re.compile(cls.PATTERNS['dimensions'], re.IGNORECASE)
        match = pattern.search(dimension_str)

        if not match:
            return result

        try:
            dim1, dim2, dim3, unit = match.groups()
            dims = [float(dim1), float(dim2), float(dim3)]
            unit = unit.lower() if unit else 'inches'

            if unit.startswith('cm'):
                factor = cls.CONVERSION_FACTORS['cm_to_inches']
                dims = [d * factor for d in dims]

            result['length'], result['width'], result['height'] = [round(d, 2) for d in dims]
        except Exception:
            pass

        return result

    def clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean all columns in the dataframe using appropriate cleaning functions."""
        # Apply cleaning functions to each column
        for col in df.columns:
            if col in self.clean_map:
                func = self.clean_map[col]
                df = df.with_columns(
                    pl.col(col).map_elements(func, return_dtype=pl.String).alias(col)
                )

        # Parse dimensions if present
        if 'dimensions' in df.columns:
            # Extract dimension components
            dim_data = df.select(
                pl.col('dimensions').map_elements(
                    self.parse_dimensions,
                    return_dtype=pl.Object
                )
            ).to_series()

            try:
                df = df.with_columns([
                    pl.Series('length', [d['length'] for d in dim_data]),
                    pl.Series('width', [d['width'] for d in dim_data]),
                    pl.Series('height', [d['height'] for d in dim_data]),
                ]).drop('dimensions')
            except Exception as e:
                print(e)

        return df

    def clean_and_save(
        self,
        df: pl.DataFrame,
        output_path: Optional[str | Path] = None,
        deduplicate_on: Optional[list[str]] = None
    ) -> pl.DataFrame:
        """Clean dataframe, optionally deduplicate and save."""
        df = self.clean_dataframe(df)

        if deduplicate_on:
            df = df.unique(subset=deduplicate_on, keep='first')

        if output_path:
            df.write_csv(output_path)

        return df


if __name__ == "__main__":
    # Load the CSV file
    df = pl.read_csv('data/amazon.csv')

    # Clean the dataframe
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_and_save(
        df=df,
        output_path='data/amazon_cleaned.csv',
        deduplicate_on=['isbn_13']
    )
    print(f"Cleaned {len(cleaned_df)} rows.")
