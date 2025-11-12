import pandas as pd
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
    def validate_text(func):
        def wrapper(self, x):
            if pd.isna(x) or (isinstance(x, str) and not x.strip()):
                return None
            if isinstance(x, str):
                return func(self, x)
            return x
        return wrapper

    @validate_text
    def clean_text(self, text: str) -> Optional[str]:
        cleaned = self.remove_invisible_chars(text)
        return cleaned if cleaned else None

    @validate_text
    def clean_isbn(self, text: str) -> Optional[str]:
        text = self.remove_invisible_chars(text)
        text = text.replace('.0', '')
        cleaned = re.sub(r'\D', '', text)
        return cleaned if cleaned else None

    @validate_text
    def clean_numeric(self, text: str) -> Optional[int]:
        match = re.search(self.PATTERNS['number'], text)
        if match:
            try:
                return int(match.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    @validate_text
    def clean_rating(self, text: str) -> Optional[float]:
        match = re.search(self.PATTERNS['rating'], text)
        if match:
            try:
                return round(float(match.group(1)), 1)
            except ValueError:
                return None
        return None

    @validate_text
    def clean_base_rating(self, text: str) -> Optional[float]:
        match = re.search(self.PATTERNS['rating_base'], text)
        if match:
            try:
                return round(float(match.group(1)), 1)
            except ValueError:
                return None
        return None

    @validate_text
    def clean_publication_date(self, text: str) -> Optional[str]:
        try:
            dt = datetime.strptime(text, '%B %d, %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return None

    @validate_text
    def clean_item_weight(self, text: str) -> Optional[float]:
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

    @validate_text
    def clean_price(self, text: str) -> Optional[float]:
        match = re.search(self.PATTERNS['price'], text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    @validate_text
    def clean_reading_age(self, text: str) -> Optional[str]:
        match = re.search(self.PATTERNS['age_range'], text)
        if match:
            return f"{match.group(1)} - {match.group(2)}"
        return None

    @staticmethod
    def remove_invisible_chars(text: str) -> str:
        return ''.join(
            c for c in text if c not in DataCleaner.INVISIBLE_CHARS and unicodedata.category(c)[0] != 'C'
        ).strip()

    @staticmethod
    def fix_numeric_series(series: pd.Series) -> pd.Series:
        def fix_value(x):
            if pd.isna(x):
                return None
            if isinstance(x, str):
                return x
            try:
                return f"{int(x)}"
            except Exception:
                return str(x)

        return series.apply(fix_value).astype('string')

    @classmethod
    def parse_dimensions(cls, dimension_str: Optional[str]) -> Dict[str, Optional[float]]:
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

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Fix numeric ID columns first
        for col in self.ID_COLUMNS:
            if col in df.columns:
                df[col] = self.fix_numeric_series(df[col])
        # Apply cleaning functions
        for col, func in self.clean_map.items():
            if col in df.columns:
                df[col] = df[col].apply(func)
        # Parse dimensions
        if 'dimensions' in df.columns:
            dims = df['dimensions'].apply(self.parse_dimensions)
            df['length'] = dims.apply(lambda d: d['length'])
            df['width'] = dims.apply(lambda d: d['width'])
            df['height'] = dims.apply(lambda d: d['height'])
            df.drop(columns=['dimensions'], inplace=True)
        return df

    def load_and_clean_csv(self, input_path: str, output_path: Optional[str] = None, deduplicate_on: Optional[list[str]] = None) -> pd.DataFrame:
        df = pd.read_csv(input_path)
        df = self.clean_dataframe(df)
        if deduplicate_on:
            df = df.drop_duplicates(subset=deduplicate_on)
        return df

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned_df = cleaner.load_and_clean_csv(
        input_path='data/amazon.csv',
        output_path='data/amazon_cleaned.csv',
        deduplicate_on=['isbn_13']
    )
    print(f"Cleaned {len(cleaned_df)} rows.")
