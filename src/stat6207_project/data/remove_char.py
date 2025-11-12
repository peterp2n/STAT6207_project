"""
Data cleaning module for Amazon product data.

This module provides functions to clean and standardize Amazon product data,
including removing invisible Unicode characters, parsing dimensions, converting
units, and standardizing date formats.
"""

import pandas as pd
import re
import unicodedata
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path


# ============================================================================
# TEXT CLEANING
# ============================================================================
INVISIBLE_CHARS = {
    '\u200b',  # Zero-width space
    '\u200c',  # Zero-width non-joiner
    '\u200d',  # Zero-width joiner
    '\u200e',  # Left-to-right mark (LRM)
    '\u200f',  # Right-to-left mark
    '\u202a',  # Left-to-right embedding
    '\u202b',  # Right-to-left embedding
    '\u202c',  # Pop directional formatting
    '\u202d',  # Left-to-right override
    '\u202e',  # Right-to-left override
    '\ufeff',  # Zero-width no-break space
    '\u00a0',  # Non-breaking space
}


def remove_invisible_chars(text: str) -> str:
    """Remove all invisible Unicode characters and control characters from text."""
    if not isinstance(text, str):
        return text
    text = ''.join(c for c in text if c not in INVISIBLE_CHARS)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    return text.strip()


# ============================================================================
# DIMENSION PARSING
# ============================================================================
def parse_dimensions(dimension_str: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Parse dimension string from Amazon and convert to inches if needed.

    FBA (Fulfillment by Amazon) Specification:
    Amazon format: a x b x c
    where:
        a = length (longest side of packaged item)
        b = width (median side of packaged item)
        c = height (shortest side of packaged item)

    Input format: "L x W x H inches" or "L x W x H cm"
    Returns dict with length, width, height in inches (None if parsing fails)
    """
    result = {'length': None, 'width': None, 'height': None}

    if not dimension_str or not isinstance(dimension_str, str):
        return result

    dimension_str = remove_invisible_chars(dimension_str)

    # Extract numbers and unit
    # Pattern: number x number x number [unit]
    pattern = r'([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)\s*(inches?|cm|centimeter|centimeters)?'
    match = re.search(pattern, dimension_str, re.IGNORECASE)

    if not match:
        return result

    try:
        dim1, dim2, dim3, unit = match.groups()
        dim1, dim2, dim3 = float(dim1), float(dim2), float(dim3)

        # Determine unit and convert to inches if needed
        unit = unit.lower() if unit else 'inches'
        conversion_factor = 1.0
        if unit.startswith('cm'):
            conversion_factor = 0.393701  # 1 cm = 0.393701 inches

        dim1 *= conversion_factor
        dim2 *= conversion_factor
        dim3 *= conversion_factor

        # FBA specification: a x b x c = length x width x height
        result['length'] = round(dim1, 2)
        result['width'] = round(dim2, 2)
        result['height'] = round(dim3, 2)

        return result
    except (ValueError, TypeError):
        return result


# ============================================================================
# FIELD-SPECIFIC CLEANING FUNCTIONS
# ============================================================================
def clean_text(text: str) -> Optional[str]:
    """Remove invisible characters from text."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    cleaned = remove_invisible_chars(text)
    return cleaned if cleaned else None


def clean_isbn(text: str) -> Optional[str]:
    """Remove hyphens and non-digit characters from ISBN."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    cleaned = re.sub(r'\D', '', text)
    return cleaned if cleaned else None


def clean_number_of_reviews(text: str) -> Optional[int]:
    """Extract number from 'X,XXX ratings' format."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    match = re.search(r'([\d,]+)', text)
    if match:
        try:
            return int(match.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def clean_rating(text: str) -> Optional[float]:
    """Extract rating score from 'X.X out of 5 stars' format."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    match = re.search(r'([\d.]+)\s+out of', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def clean_customer_reviews(text: str) -> Optional[float]:
    """Extract rating number from 'X out of 5' format."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    match = re.search(r'([\d.]+)\s+out of 5', text)
    if match:
        try:
            rating = float(match.group(1))
            return round(rating, 1)
        except ValueError:
            return None
    return None


def clean_print_length(text: str) -> Optional[int]:
    """Extract page number from 'X pages' format, handling comma-separated numbers."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    match = re.search(r'([\d,]+)', text)
    if match:
        try:
            return int(match.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def clean_publication_date(text: str) -> Optional[str]:
    """Convert publication date from 'Month D, YYYY' format to 'YYYY-MM-DD' format."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    try:
        date_obj = datetime.strptime(text, '%B %d, %Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None


def clean_item_weight(text: str) -> Optional[float]:
    """Convert item weight to grams. Handles 'X pounds, Y ounces' format."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    try:
        # Extract pounds and ounces values
        pounds_match = re.search(r'([\d.]+)\s*pounds?', text)
        ounces_match = re.search(r'([\d.]+)\s*ounces?', text)
        total_grams = 0.0

        if pounds_match:
            pounds = float(pounds_match.group(1))
            total_grams += pounds * 453.592  # 1 pound = 453.592 grams

        if ounces_match:
            ounces = float(ounces_match.group(1))
            total_grams += ounces * 28.3495  # 1 ounce = 28.3495 grams

        if total_grams > 0:
            return round(total_grams, 2)
        return None
    except (ValueError, AttributeError):
        return None


def clean_price(text: str) -> Optional[float]:
    """Extract price value from price string."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    # Extract numeric value, handling comma as thousands separator
    match = re.search(r'([\d,]+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def clean_reading_age(text: str) -> Optional[str]:
    """Extract only the 'number - number' age range pattern from reading age string."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)

    # Pattern to match: digits, hyphen, digits (with optional spaces)
    match = re.search(r'(\d+)\s*-\s*(\d+)', text)

    if match:
        # Return the pattern as "number - number" with consistent spacing
        return f"{match.group(1)} - {match.group(2)}"
    return None


# ============================================================================
# DATAFRAME CLEANING
# ============================================================================
def clean_amazon_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Amazon product data DataFrame.

    Applies field-specific cleaning functions to each column and handles
    dimension parsing to split dimensions into length, width, height.

    Args:
        df: Pandas DataFrame with raw Amazon product data

    Returns:
        Cleaned Pandas DataFrame with standardized columns
    """

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Apply text cleaning to all string columns first
    text_columns = [
        'asin', 'author', 'availability', 'barcode', 'best_sellers_rank',
        'book_format', 'description', 'edition', 'features', 'language',
        'part_of_series', 'product_name', 'product_url', 'publisher'
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Clean ISBN columns
    if 'isbn_10' in df.columns:
        df['isbn_10'] = df['isbn_10'].apply(clean_isbn)

    if 'isbn_13' in df.columns:
        df['isbn_13'] = df['isbn_13'].apply(clean_isbn)

    if 'barcode' in df.columns:
        df['barcode'] = df['barcode'].apply(clean_isbn)

    # Clean numeric review count
    if 'number_of_reviews' in df.columns:
        df['number_of_reviews'] = df['number_of_reviews'].apply(clean_number_of_reviews)

    # Clean ratings (convert to float)
    if 'rating' in df.columns:
        df['rating'] = df['rating'].apply(clean_rating)

    if 'customer_reviews' in df.columns:
        df['customer_reviews'] = df['customer_reviews'].apply(clean_customer_reviews)

    # Clean print length (convert to integer pages)
    if 'print_length' in df.columns:
        df['print_length'] = df['print_length'].apply(clean_print_length)

    # Clean publication date (convert to YYYY-MM-DD)
    if 'publication_date' in df.columns:
        df['publication_date'] = df['publication_date'].apply(clean_publication_date)

    # Clean item weight (convert to grams)
    if 'item_weight' in df.columns:
        df['item_weight'] = df['item_weight'].apply(clean_item_weight)

    # Clean price (convert to float)
    if 'price' in df.columns:
        df['price'] = df['price'].apply(clean_price)

    # Clean reading age (extract only "X - Y" pattern)
    if 'reading_age' in df.columns:
        df['reading_age'] = df['reading_age'].apply(clean_reading_age)

    # Parse dimensions into separate columns
    if 'dimensions' in df.columns:
        # Parse dimensions and create new columns
        dimensions_parsed = df['dimensions'].apply(parse_dimensions)

        # Extract length, width, height from parsed dimensions
        df['length'] = dimensions_parsed.apply(lambda d: d['length'])
        df['width'] = dimensions_parsed.apply(lambda d: d['width'])
        df['height'] = dimensions_parsed.apply(lambda d: d['height'])

        # Drop the original dimensions column
        df = df.drop('dimensions', axis=1)

    return df


def clean_amazon_csv(
    input_path: str = 'data/amazon.csv',
    output_path: str = 'data/amazon_cleaned.csv'
) -> pd.DataFrame:
    """
    Load, clean, and save Amazon product data.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file

    Returns:
        Cleaned Pandas DataFrame
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file.resolve()}")

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    print("Cleaning data...")
    # Drop duplicated rows based on 'isbn_13' after cleaning
    df_cleaned = clean_amazon_dataframe(df).drop_duplicates(subset=["isbn_13"])

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving cleaned data to {output_path}...")
    df_cleaned.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df_cleaned)} rows to {output_path}")

    return df_cleaned


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    try:
        df_cleaned = clean_amazon_csv(
            input_path='data/amazon.csv',
            output_path='data/amazon_cleaned.csv'
        )

        print(f"\n{'=' * 60}")
        print("Data cleaning completed!")
        print(f"  Rows: {len(df_cleaned)}")
        print(f"  Columns: {len(df_cleaned.columns)}")
        print(f"  Column names: {', '.join(df_cleaned.columns)}")
        print(f"{'=' * 60}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error cleaning data: {e}")
        raise


if __name__ == "__main__":
    main()
