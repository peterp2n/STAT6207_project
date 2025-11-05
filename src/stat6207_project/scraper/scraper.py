import polars as pl
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
import time
import csv
import os
import re
import unicodedata
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

# Predefined CSV column order
# Dimensions ordered per FBA (Fulfillment by Amazon) specification:
# - length: longest side (a)
# - width: median side (b)
# - height: shortest side (c)
CSV_FIELDNAMES = [
    'isbn', 'product_name', 'asin', 'author', 'availability', 'best_sellers_rank',
    'customer_reviews', 'description', 'edition', 'error_message', 'features',
    'isbn_10', 'isbn_13', 'item_weight', 'language', 'number_of_reviews',
    'part_of_series', 'price', 'print_length', 'product_url', 'publication_date',
    'publisher', 'rating', 'reading_age', 'length', 'width', 'height', 'scrape_status'
]


def clean_data(product_data: Dict) -> Dict:
    """Clean product data by removing invisible Unicode characters from text fields."""

    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return text
        text = ''.join(c for c in text if c not in INVISIBLE_CHARS)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return ''.join(c for c in text if unicodedata.category(c)[0] != 'C').strip()

    def _clean_isbn(text: str) -> Optional[str]:
        """Remove hyphens and non-digit characters from ISBN."""
        if not isinstance(text, str):
            return text
        return re.sub(r'\D', '', text)

    def _clean_number_of_reviews(text: str) -> Optional[str]:
        """Extract number from 'X,XXX ratings' format."""
        if not isinstance(text, str):
            return text

        match = re.search(r'([\d,]+)', text)
        if match:
            return match.group(1).replace(',', '')
        return text

    def _clean_customer_reviews(text: str) -> Optional[str]:
        """Extract rating number from 'X out of 5' format, keeping integer or 1 dp."""
        if not isinstance(text, str):
            return text

        match = re.search(r'([\d.]+)\s+out of 5', text)
        if match:
            rating = float(match.group(1))
            rating = round(rating, 1)
            return str(int(rating)) if rating == int(rating) else f"{rating:.1f}"
        return text

    clean = {}
    for k, v in product_data.items():
        if k in ('isbn_10', 'isbn_13') and isinstance(v, str):
            clean[k] = _clean_isbn(v)
        elif k == 'number_of_reviews' and isinstance(v, str):
            clean[k] = _clean_number_of_reviews(v)
        elif k == 'customer_reviews' and isinstance(v, str):
            clean[k] = _clean_customer_reviews(v)
        elif isinstance(v, str):
            clean[k] = _clean_text(v)
        else:
            clean[k] = v

    return clean


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

    dimension_str = dimension_str.strip()

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
        # a (dim1) = length (longest side)
        # b (dim2) = width (median side)
        # c (dim3) = height (shortest side)
        result['length'] = round(dim1, 2)
        result['width'] = round(dim2, 2)
        result['height'] = round(dim3, 2)

        return result

    except (ValueError, TypeError):
        return result


# ============================================================================
# CSV FILE OPERATIONS
# ============================================================================

def append_record_to_csv(record: Dict, filename: str = 'data/scrape_results.csv') -> None:
    """
    Append a single record to CSV file using predefined column order.
    Create parent directory if it doesn't exist.
    Uses key-based matching instead of positional ordering.

    Dimension columns follow FBA (Fulfillment by Amazon) specification:
    - length: longest side of packaged item
    - width: median side of packaged item
    - height: shortest side of packaged item
    """
    csv_path = Path(filename)

    # Create parent directory if it doesn't exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, restval='')

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        # Write row using key-based matching
        writer.writerow(record)


def get_already_scraped_isbns(csv_filename: str = 'data/scrape_results.csv') -> list:
    """Extract already scraped ISBNs from CSV if it exists."""
    csv_path = Path(csv_filename)

    if not csv_path.exists():
        print(f"No existing CSV file found. Starting fresh scrape.")
        return []

    try:
        df = pl.read_csv(csv_path)

        if 'isbn' not in df.columns:
            print("Warning: 'isbn' column not found in CSV")
            return []

        isbn_col = df.select('isbn').to_series().to_list()
        print(f"Found {len(isbn_col)} already scraped records in {csv_filename}")
        return isbn_col

    except Exception as e:
        print(f"Error reading existing CSV: {e}")
        return []


def filter_barcodes_to_scrape(barcodes: list, csv_filename: str = 'data/scrape_results.csv') -> list:
    """Filter out barcodes that have already been scraped."""
    isbn_col = get_already_scraped_isbns(csv_filename)

    if not isbn_col:
        return barcodes

    mask = ~np.isin(barcodes, isbn_col)
    filtered_barcodes = np.array(barcodes)[mask].tolist()

    skipped_count = len(barcodes) - len(filtered_barcodes)
    print(f"Skipping {skipped_count} already-scraped records")
    print(f"Will scrape {len(filtered_barcodes)} new records")

    return filtered_barcodes


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def create_sqlite_engine(db_path: str):
    """Create SQLAlchemy engine for SQLite database."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_file.resolve()}")

    engine = create_engine(f"sqlite:///{db_file}")
    print(f"Connected to database: {db_file.resolve()}")
    return engine


def load_table_from_sqlite(engine, table_name: str) -> pl.DataFrame:
    """Load a table from SQLite database using SQLAlchemy engine."""
    query = f"SELECT * FROM {table_name}"
    with engine.connect() as conn:
        df = pl.read_database(query=query, connection=conn)
    print(f"Loaded {len(df)} rows from table '{table_name}'")
    return df


def load_all_tables(db_path: str) -> Dict[str, pl.DataFrame]:
    """Load all tables from the database using SQLAlchemy."""
    engine = create_sqlite_engine(db_path)
    table_names = ['products', 'purchase', 'sales', 'shops']
    return {table_name: load_table_from_sqlite(engine, table_name)
            for table_name in table_names}


def extract_barcodes(products_df: pl.DataFrame, column_name: str = 'barcode2') -> list:
    """Extract unique barcodes from products DataFrame."""
    barcodes = (
        products_df
        .select(column_name)
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )
    print(f"Found {len(barcodes)} unique barcodes in database")
    return barcodes


# ============================================================================
# WEBDRIVER SETUP
# ============================================================================

def initialize_driver(headless: bool = True) -> webdriver.Chrome:
    """Initialize Chrome WebDriver with anti-detection settings."""
    options = webdriver.ChromeOptions()

    if headless:
        options.add_argument('--headless')
        print("Running in headless mode (no browser window)")

    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


# ============================================================================
# AMAZON NAVIGATION
# ============================================================================

def search_amazon(barcode: str, driver: webdriver.Chrome) -> Optional[str]:
    """Search Amazon for barcode and return URL of first product."""
    search_url = f"https://www.amazon.com/s?k={barcode}&language=en_US&currency=HKD"
    driver.get(search_url)
    time.sleep(3)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot"))
        )
        time.sleep(2)

        product_links = [
            link for link in driver.find_elements(By.CSS_SELECTOR, "div.s-result-item a")
            if (href := link.get_attribute('href')) and '/dp/' in href and link.text.strip()
        ]

        if not product_links:
            print(f"✗ No search results found for isbn: {barcode}")
            return None

        product_url = product_links[0].get_attribute('href')
        print(f"  Found product: {product_links[0].text.strip()[:60]}...")
        return product_url

    except TimeoutException:
        print(f"✗ No results found for isbn: {barcode}")
        return None


def navigate_to_product(product_url: str, driver: webdriver.Chrome) -> None:
    """Navigate to product page."""
    driver.get(product_url)
    time.sleep(4)


# ============================================================================
# HTML PARSING AND DATA EXTRACTION
# ============================================================================

EMPTY_PRODUCT = {
    'isbn': None, 'product_url': None, 'product_name': None, 'price': None,
    'rating': None, 'number_of_reviews': None, 'availability': None,
    'description': None, 'features': None, 'isbn_10': None, 'isbn_13': None,
    'publisher': None, 'publication_date': None, 'language': None,
    'length': None, 'width': None, 'height': None,
    'item_weight': None, 'print_length': None,
    'reading_age': None, 'edition': None, 'author': None, 'asin': None,
    'part_of_series': None, 'best_sellers_rank': None, 'customer_reviews': None,
    'scrape_status': None, 'error_message': None
}

FIELD_MAPPINGS = {
    'ISBN-10': 'isbn_10', 'ISBN-13': 'isbn_13', 'Publisher': 'publisher',
    'Publication date': 'publication_date', 'Language': 'language',
    'Dimensions': 'dimensions',
    'Item Weight': 'item_weight',
    'Item weight': 'item_weight', 'Print length': 'print_length',
    'Paperback': 'print_length', 'Reading age': 'reading_age',
    'Edition': 'edition', 'ASIN': 'asin', 'Series': 'part_of_series',
    'Part of': 'part_of_series', 'Best Sellers Rank': 'best_sellers_rank'
}


def create_product_dict(barcode: str, url: str = None) -> Dict:
    """Create product data dictionary."""
    product = EMPTY_PRODUCT.copy()
    product['isbn'] = barcode
    product['product_url'] = url
    return product


def create_failed_product(barcode: str, error: str) -> Dict:
    """Create a failed product record with error message."""
    product = create_product_dict(barcode)
    product['scrape_status'] = 'FAILED'
    product['error_message'] = error
    return product


def extract_basic_info(soup: BeautifulSoup, product_data: Dict) -> None:
    """Extract basic product information."""
    elem_specs = [
        ('productTitle', 'span', 'id', 'product_name'),
        ('a-price-whole', 'span', 'class', 'price'),
        ('a-icon-alt', 'span', 'class', 'rating'),
        ('acrCustomerReviewText', 'span', 'id', 'number_of_reviews'),
        ('availability', 'div', 'id', 'availability'),
    ]

    for elem_id, tag, attr, field in elem_specs:
        if elem := soup.find(tag, {attr: elem_id}):
            product_data[field] = elem.text.strip()

    if reviews_rating := soup.find('span', {'data-hook': 'rating-out-of-text'}):
        product_data['customer_reviews'] = reviews_rating.text.strip()


def extract_description(soup: BeautifulSoup, product_data: Dict) -> None:
    """Extract product description."""
    if desc_elem := soup.find('div', {'id': 'bookDescription_feature_div'}):
        product_data['description'] = (
            desc_elem.find('span').text.strip() if desc_elem.find('span')
            else desc_elem.text.strip()
        )
    elif not product_data['description']:
        if feature_div := soup.find('div', {'id': 'feature-bullets'}):
            product_data['description'] = feature_div.text.strip()


def extract_author(soup: BeautifulSoup, product_data: Dict) -> None:
    """Extract author information."""
    if author_elem := soup.find('span', {'class': 'author'}):
        if author_link := author_elem.find('a', {'class': 'a-link-normal'}):
            product_data['author'] = author_link.text.strip()


def extract_features(soup: BeautifulSoup, product_data: Dict) -> None:
    """Extract product feature bullets."""
    if feature_bullets := soup.find('div', {'id': 'feature-bullets'}):
        features = [
            bullet.text.strip() for bullet in feature_bullets.find_all('span', {'class': 'a-list-item'})
            if bullet.text.strip()
        ]
        product_data['features'] = ' | '.join(features) if features else None


def parse_detail_field(text: str, product_data: Dict) -> None:
    """Parse and update a single detail field."""
    for key, field_name in FIELD_MAPPINGS.items():
        if key in text:
            value = text.split(':')[-1].strip()

            # Special handling for Dimensions
            if field_name == 'dimensions':
                dim_dict = parse_dimensions(value)
                product_data['length'] = dim_dict['length']
                product_data['width'] = dim_dict['width']
                product_data['height'] = dim_dict['height']
            else:
                product_data[field_name] = value
            return


def extract_details_from_bullets(soup: BeautifulSoup, product_data: Dict) -> None:
    """Extract product details from bullet list format."""
    if details_section := soup.find('div', {'id': 'detailBullets_feature_div'}):
        for detail in details_section.find_all('li'):
            parse_detail_field(detail.text.strip(), product_data)


def extract_details_from_table(soup: BeautifulSoup, product_data: Dict) -> None:
    """Extract product details from table format."""
    if details_table := soup.find('table', {'id': 'productDetails_detailBullets_sections1'}):
        for row in details_table.find_all('tr'):
            if (header := row.find('th')) and (value := row.find('td')):
                parse_detail_field(f"{header.text.strip()}: {value.text.strip()}", product_data)


def extract_all_product_data(soup: BeautifulSoup, barcode: str, url: str) -> Dict:
    """Extract all product data from parsed HTML."""
    product_data = create_product_dict(barcode, url)

    extract_basic_info(soup, product_data)
    extract_description(soup, product_data)
    extract_author(soup, product_data)
    extract_features(soup, product_data)
    extract_details_from_bullets(soup, product_data)
    extract_details_from_table(soup, product_data)

    product_data['scrape_status'] = 'SUCCESS'
    return clean_data(product_data)


# ============================================================================
# MAIN SCRAPING ORCHESTRATION
# ============================================================================

def scrape_amazon_product(barcode: str, driver: webdriver.Chrome,
                          csv_filename: str = 'data/scrape_results.csv') -> bool:
    """Search Amazon for a barcode, scrape data, and append to CSV (including failures)."""
    try:
        if not (product_url := search_amazon(barcode, driver)):
            failed_product = create_failed_product(barcode, 'No search results found')
            append_record_to_csv(failed_product, csv_filename)
            print(f"⚠ Saved failed record for isbn: {barcode}")
            return False

        navigate_to_product(product_url, driver)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        product_data = extract_all_product_data(soup, barcode, driver.current_url)

        append_record_to_csv(product_data, csv_filename)
        print(f"✓ Successfully scraped and saved: {barcode}")
        return True

    except TimeoutException as e:
        failed_product = create_failed_product(barcode, f'Timeout: {str(e)}')
        append_record_to_csv(failed_product, csv_filename)
        print(f"⚠ Timeout for isbn {barcode} - saved partial record")
        return False

    except Exception as e:
        error_msg = f'{type(e).__name__}: {str(e)[:100]}'
        failed_product = create_failed_product(barcode, error_msg)
        append_record_to_csv(failed_product, csv_filename)
        print(f"✗ Error scraping isbn {barcode}: {error_msg}")
        return False


def process_barcodes(barcodes: list, driver: webdriver.Chrome,
                     csv_filename: str = 'data/scrape_results.csv') -> tuple:
    """Process list of barcodes and append each record to CSV individually."""
    successful_count = 0
    failed_count = 0

    for idx, barcode in enumerate(barcodes, 1):
        print(f"\n[{idx}/{len(barcodes)}] Processing isbn: {barcode}")

        if scrape_amazon_product(barcode, driver, csv_filename):
            successful_count += 1
        else:
            failed_count += 1

        time.sleep(4)

    return successful_count, failed_count


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        print("Loading tables from database...")
        tables = load_all_tables('Topic1_dataset.sqlite')
        barcodes = extract_barcodes(tables['products'], column_name='barcode2')

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"Current working directory: {os.getcwd()}")
        return
    except Exception as e:
        print(f"\n❌ Error loading database: {e}")
        return

    output_file = 'data/scrape_results.csv'

    barcodes_to_scrape = filter_barcodes_to_scrape(barcodes, output_file)[:5]

    if not barcodes_to_scrape:
        print("\n✓ All records have already been scraped!")
        return

    print("\nInitializing Chrome WebDriver...")
    driver = initialize_driver(headless=True)

    try:
        successful_count, failed_count = process_barcodes(barcodes_to_scrape, driver, output_file)

        total_count = successful_count + failed_count
        print(f"\n{'=' * 60}")
        print(f"Scraping session completed!")
        print(f"  ✓ Successful: {successful_count}")
        print(f"  ✗ Failed: {failed_count}")
        print(f"  Session total: {total_count}")
        print(f"  Saved to: {output_file}")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping interrupted by user")
        print(f"✓ Partial results saved to {output_file}")

    finally:
        print("\nClosing browser...")
        driver.quit()
        print("Done!")


if __name__ == "__main__":
    main()
