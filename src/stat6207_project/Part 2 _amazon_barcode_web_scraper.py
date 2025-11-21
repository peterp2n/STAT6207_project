"""
Amazon Product Scraper for STAT6207 Final Project
Description extraction enhancements:
- Extracts from collapsed expander divs (data-expanded="false")
- Clicks "Read more" button to expand full descriptions
- Removes "Read more"/"Read less" text artifacts (English and Chinese)
- Handles multi-span descriptions
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import csv
import re
import unicodedata
from datetime import datetime
from typing import Optional, Dict


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================
INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
    '\ufeff', '\u00a0',
}


def remove_invisible_chars(text: str) -> str:
    """Remove invisible Unicode characters and control characters."""
    if not isinstance(text, str):
        return text
    text = ''.join(c for c in text if c not in INVISIBLE_CHARS)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    return text.strip()


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


def clean_print_length(text: str) -> Optional[int]:
    """Extract page number from 'X pages' format."""
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
    """Convert date from 'Month D, YYYY' to 'YYYY-MM-DD' format."""
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
    """Convert weight to grams from pounds/ounces."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    try:
        pounds_match = re.search(r'([\d.]+)\s*pounds?', text)
        ounces_match = re.search(r'([\d.]+)\s*ounces?', text)
        total_grams = 0.0
        if pounds_match:
            total_grams += float(pounds_match.group(1)) * 453.592
        if ounces_match:
            total_grams += float(ounces_match.group(1)) * 28.3495
        if total_grams > 0:
            return round(total_grams, 2)
        return None
    except (ValueError, AttributeError):
        return None


def clean_price(text: str) -> Optional[float]:
    """Extract numeric price value."""
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return None
    if not isinstance(text, str):
        return text
    text = remove_invisible_chars(text)
    match = re.search(r'([\d,]+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def parse_dimensions(dimension_str: Optional[str]) -> Dict[str, Optional[float]]:
    """Parse dimensions and convert to inches."""
    result = {'length': None, 'width': None, 'height': None}
    if not dimension_str or not isinstance(dimension_str, str):
        return result
    dimension_str = remove_invisible_chars(dimension_str)
    pattern = r'([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)\s*(inches?|cm)?'
    match = re.search(pattern, dimension_str, re.IGNORECASE)
    if not match:
        return result
    try:
        dim1, dim2, dim3, unit = match.groups()
        dim1, dim2, dim3 = float(dim1), float(dim2), float(dim3)
        unit = unit.lower() if unit else 'inches'
        conversion_factor = 0.393701 if unit.startswith('cm') else 1.0
        result['length'] = round(dim1 * conversion_factor, 2)
        result['width'] = round(dim2 * conversion_factor, 2)
        result['height'] = round(dim3 * conversion_factor, 2)
        return result
    except (ValueError, TypeError):
        return result


def clean_amazon_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all columns in the DataFrame."""
    df = df.copy()
    
    # Clean text columns
    text_cols = ['asin', 'author', 'availability', 'barcode', 'best_sellers_rank',
                 'book_format', 'description', 'edition', 'features', 'language',
                 'part_of_series', 'product_name', 'product_url', 'publisher', 
                 'reading_age']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Clean ISBN columns
    for col in ['isbn_10', 'isbn_13', 'barcode']:
        if col in df.columns:
            df[col] = df[col].apply(clean_isbn)
    
    # Clean numeric columns
    if 'number_of_reviews' in df.columns:
        df['number_of_reviews'] = df['number_of_reviews'].apply(clean_number_of_reviews)
    if 'rating' in df.columns:
        df['rating'] = df['rating'].apply(clean_rating)
    if 'customer_reviews' in df.columns:
        df['customer_reviews'] = df['rating'].apply(clean_rating)
    if 'print_length' in df.columns:
        df['print_length'] = df['print_length'].apply(clean_print_length)
    if 'publication_date' in df.columns:
        df['publication_date'] = df['publication_date'].apply(clean_publication_date)
    if 'item_weight' in df.columns:
        df['item_weight'] = df['item_weight'].apply(clean_item_weight)
    if 'price' in df.columns:
        df['price'] = df['price'].apply(clean_price)
    
    # Parse dimensions
    if 'dimensions' in df.columns:
        dimensions_parsed = df['dimensions'].apply(parse_dimensions)
        df['length'] = dimensions_parsed.apply(lambda d: d['length'])
        df['width'] = dimensions_parsed.apply(lambda d: d['width'])
        df['height'] = dimensions_parsed.apply(lambda d: d['height'])
        df = df.drop('dimensions', axis=1)
    
    return df


# ============================================================================
# WEB SCRAPING FUNCTIONS
# ============================================================================
def scrape_amazon_product(barcode, driver):
    """
    Search Amazon for a barcode and scrape the first product result
    
    Args:
        barcode: The barcode to search for
        driver: Selenium WebDriver instance
    
    Returns:
        dict: Product details or None if no results found
    """
    try:
        # Navigate to Amazon search with barcode (English language, USD currency)
        search_url = f"https://www.amazon.com/s?k={barcode}&language=en_US&currency=USD"
        driver.get(search_url)
        
        # Wait for page to load completely (MAXIMUM SPEED)
        time.sleep(0.3)
        
        # Try to find the first product link with multiple selectors
        try:
            # Wait for search results to appear
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot"))
            )
            
            time.sleep(0.1)
            
            # Find all links in search results that point to product pages
            all_links = driver.find_elements(By.CSS_SELECTOR, "div.s-result-item a")
            
            # Filter for actual product links (containing /dp/)
            product_links = []
            for link in all_links:
                href = link.get_attribute('href')
                if href and '/dp/' in href and link.text.strip():
                    product_links.append(link)
            
            if not product_links or len(product_links) == 0:
                print(f"✗ No search results found for barcode: {barcode}")
                return None
            
            # Get the first product link
            first_product = product_links[0]
            product_url = first_product.get_attribute('href')
            print(f"  Found product: {first_product.text.strip()[:60]}...")
            
            # Scroll to element to make sure it's visible
            driver.execute_script("arguments[0].scrollIntoView(true);", first_product)
            
            # Click on the first product title using JavaScript to ensure it works
            driver.execute_script("arguments[0].click();", first_product)
            
            # Wait for product page to load (MAXIMUM SPEED)
            time.sleep(0.5)
            
            # Get page source and parse with BeautifulSoup
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract product details
            product_data = {
                'barcode': barcode,
                'product_url': driver.current_url,
                'product_name': None,
                'book_format': None,  # NEW: Paperback, Hardcover, Kindle, etc.
                'price': None,
                'rating': None,
                'number_of_reviews': None,
                'availability': None,
                'description': None,
                'features': None,
                'isbn_10': None,
                'isbn_13': None,
                'publisher': None,
                'publication_date': None,
                'language': None,
                'dimensions': None,
                'item_weight': None,
                'print_length': None,
                'reading_age': None,
                'edition': None,
                'author': None,
                'asin': None,
                'part_of_series': None,
                'best_sellers_rank': None,
                'customer_reviews': None
            }
            
            # Extract product name (title)
            title_elem = soup.find('span', {'id': 'productTitle'})
            if title_elem:
                product_data['product_name'] = title_elem.text.strip()
            
            # Detect book format (Paperback, Hardcover, Kindle, etc.)
            book_format = "Unknown"
            
            # Method 1: Check binding information from product details
            format_elem = soup.find('a', {'id': 'a-autoid-4-announce'})
            if format_elem:
                format_text = format_elem.text.strip().lower()
                if 'paperback' in format_text:
                    book_format = "Paperback"
                elif 'hardcover' in format_text or 'hardback' in format_text:
                    book_format = "Hardcover"
                elif 'kindle' in format_text:
                    book_format = "Kindle"
            
            # Method 2: Check from title
            if book_format == "Unknown" and product_data['product_name']:
                title_lower = product_data['product_name'].lower()
                if 'paperback' in title_lower:
                    book_format = "Paperback"
                elif 'hardcover' in title_lower or 'hardback' in title_lower:
                    book_format = "Hardcover"
                elif 'kindle' in title_lower:
                    book_format = "Kindle"
            
            # Method 3: Check from format buttons/selectors
            if book_format == "Unknown":
                format_buttons = soup.find_all('span', {'class': 'a-button-text'})
                for button in format_buttons:
                    button_text = button.text.strip().lower()
                    if 'selected' in button.get('class', []) or \
                       button.find_parent('a', {'class': 'a-button-selected'}):
                        if 'paperback' in button_text:
                            book_format = "Paperback"
                        elif 'hardcover' in button_text or 'hardback' in button_text:
                            book_format = "Hardcover"
                        elif 'kindle' in button_text:
                            book_format = "Kindle"
                        break
            
            product_data['book_format'] = book_format
            if book_format != "Unknown":
                print(f"  📚 Format: {book_format}")
            
            # Check currency symbol (USD only)
            currency_symbol = soup.find('span', {'class': 'a-price-symbol'})
            if currency_symbol:
                symbol_text = currency_symbol.text.strip()
                if symbol_text not in ['$', 'US$', 'USD']:
                    print(f"  ⚠ Non-USD price detected ({symbol_text}), skipping...")
            
            # Extract price (whole + fraction)
            price_whole = soup.find('span', {'class': 'a-price-whole'})
            price_fraction = soup.find('span', {'class': 'a-price-fraction'})
            if price_whole:
                whole = price_whole.text.strip()
                fraction_text = price_fraction.text.strip() if price_fraction else '00'
                full_price = f"{whole}{fraction_text}"
                
                # Validate price range (USD typically < $500 for books)
                try:
                    price_val = float(full_price.replace(',', ''))
                    if price_val > 500:
                        print(f"  ⚠ Suspicious price (${price_val}), likely HKD")
                        full_price = None
                except:
                    pass
                
                product_data['price'] = full_price
            
            # Extract rating
            rating_elem = soup.find('span', {'class': 'a-icon-alt'})
            if rating_elem:
                product_data['rating'] = rating_elem.text.strip()
            
            # Extract number of reviews
            reviews_elem = soup.find('span', {'id': 'acrCustomerReviewText'})
            if reviews_elem:
                product_data['number_of_reviews'] = reviews_elem.text.strip()
            
            # Extract availability
            availability_elem = soup.find('div', {'id': 'availability'})
            if availability_elem:
                product_data['availability'] = availability_elem.text.strip()
            
            # Extract description (from book description section)
            # Try to click "Read more" if it exists to expand full description
            try:
                read_more_button = driver.find_element(By.ID, 'bookDesc_readmore_label')
                if read_more_button:
                    driver.execute_script("arguments[0].click();", read_more_button)
                    time.sleep(0.2)  # Wait for content to expand
                    # Re-parse the page after expanding
                    page_source = driver.page_source
                    soup = BeautifulSoup(page_source, 'html.parser')
            except:
                pass  # No read more button, continue
            
            # Now extract the full description
            description_elem = soup.find('div', {'id': 'bookDescription_feature_div'})
            if description_elem:
                # Get all text from all spans (some descriptions split across multiple)
                all_spans = description_elem.find_all('span')
                desc_parts = []
                for span in all_spans:
                    text = span.text.strip()
                    if text and text not in desc_parts:
                        desc_parts.append(text)
                
                # ALSO extract from collapsed expander divs (data-expanded="false")
                # These contain the full hidden content before "Read more" is clicked
                expander_divs = description_elem.find_all('div', {'class': 'a-expander-content'})
                for div in expander_divs:
                    text = div.get_text(separator=' ', strip=True)
                    # Only add if it's substantial content (not just "Read more")
                    if text and len(text) > 20 and text not in desc_parts:
                        desc_parts.append(text)
                
                if desc_parts:
                    # Join all parts and clean up
                    full_desc = ' '.join(desc_parts)
                    # Remove common truncation artifacts (English and Chinese)
                    full_desc = full_desc.replace('Read more', '').replace('Read less', '').strip()
                    full_desc = full_desc.replace('閱讀更多', '').replace('阅读更多', '').strip()
                    product_data['description'] = full_desc
                else:
                    product_data['description'] = description_elem.text.strip()
            
            # If no book description, try feature bullets
            if not product_data['description']:
                feature_div = soup.find('div', {'id': 'feature-bullets'})
                if feature_div:
                    product_data['description'] = feature_div.text.strip()
            
            # Extract product details from the details section
            details_section = soup.find('div', {'id': 'detailBullets_feature_div'})
            if details_section:
                details_list = details_section.find_all('li')
                for detail in details_list:
                    text = detail.text.strip()
                    if 'ISBN-10' in text:
                        product_data['isbn_10'] = text.split(':')[-1].strip()
                    elif 'ISBN-13' in text:
                        product_data['isbn_13'] = text.split(':')[-1].strip()
                    elif 'Publisher' in text:
                        product_data['publisher'] = text.split(':')[-1].strip()
                    elif 'Publication date' in text:
                        product_data['publication_date'] = text.split(':')[-1].strip()
                    elif 'Language' in text:
                        product_data['language'] = text.split(':')[-1].strip()
                    elif 'Dimensions' in text:
                        product_data['dimensions'] = text.split(':')[-1].strip()
                    elif 'Item Weight' in text or 'Item weight' in text:
                        product_data['item_weight'] = text.split(':')[-1].strip()
                    elif 'Print length' in text or 'Paperback' in text:
                        product_data['print_length'] = text.split(':')[-1].strip()
                    elif 'Reading age' in text:
                        product_data['reading_age'] = text.split(':')[-1].strip()
                    elif 'Edition' in text:
                        product_data['edition'] = text.split(':')[-1].strip()
                    elif 'ASIN' in text:
                        product_data['asin'] = text.split(':')[-1].strip()
                    elif 'Series' in text or 'Part of' in text:
                        product_data['part_of_series'] = text.split(':')[-1].strip()
                    elif 'Best Sellers Rank' in text:
                        product_data['best_sellers_rank'] = text.split(':')[-1].strip()
            
            # Try alternative details table format
            if not details_section:
                details_table = soup.find('table', {'id': 'productDetails_detailBullets_sections1'})
                if details_table:
                    rows = details_table.find_all('tr')
                    for row in rows:
                        header = row.find('th')
                        value = row.find('td')
                        if header and value:
                            header_text = header.text.strip()
                            value_text = value.text.strip()
                            if 'ISBN-10' in header_text:
                                product_data['isbn_10'] = value_text
                            elif 'ISBN-13' in header_text:
                                product_data['isbn_13'] = value_text
                            elif 'Publisher' in header_text:
                                product_data['publisher'] = value_text
                            elif 'Publication date' in header_text:
                                product_data['publication_date'] = value_text
                            elif 'Language' in header_text:
                                product_data['language'] = value_text
                            elif 'Dimensions' in header_text:
                                product_data['dimensions'] = value_text
                            elif 'Item Weight' in header_text or 'Item weight' in header_text:
                                product_data['item_weight'] = value_text
                            elif 'Print length' in header_text or 'Paperback' in header_text:
                                product_data['print_length'] = value_text
                            elif 'Reading age' in header_text:
                                product_data['reading_age'] = value_text
                            elif 'Edition' in header_text:
                                product_data['edition'] = value_text
                            elif 'ASIN' in header_text:
                                product_data['asin'] = value_text
                            elif 'Series' in header_text or 'Part of' in header_text:
                                product_data['part_of_series'] = value_text
                            elif 'Best Sellers Rank' in header_text:
                                product_data['best_sellers_rank'] = value_text
            
            # Extract customer reviews rating
            reviews_rating = soup.find('span', {'data-hook': 'rating-out-of-text'})
            if reviews_rating:
                product_data['customer_reviews'] = reviews_rating.text.strip()
            
            # Extract author
            author_elem = soup.find('span', {'class': 'author'})
            if author_elem:
                author_link = author_elem.find('a', {'class': 'a-link-normal'})
                if author_link:
                    product_data['author'] = author_link.text.strip()
            
            # Extract feature bullets
            features_list = []
            feature_bullets = soup.find('div', {'id': 'feature-bullets'})
            if feature_bullets:
                bullets = feature_bullets.find_all('span', {'class': 'a-list-item'})
                for bullet in bullets:
                    text = bullet.text.strip()
                    if text:
                        features_list.append(text)
            product_data['features'] = ' | '.join(features_list) if features_list else None
            
            print(f"✓ Successfully scraped data for barcode: {barcode}")
            return product_data
            
        except TimeoutException:
            print(f"✗ No results found for barcode: {barcode}")
            return None
        except NoSuchElementException:
            print(f"✗ Could not find product elements for barcode: {barcode}")
            return None
            
    except Exception as e:
        print(f"✗ Error scraping barcode {barcode}: {str(e)}")
        return None

def main():
    # Read the products.csv file
    print("Reading products.csv file...")
    df = pd.read_csv('products.csv')
    
    # Extract barcode2 column
    barcodes = df['barcode2'].dropna().unique().tolist()
    print(f"Found {len(barcodes)} unique barcodes to process")
    
    # Initialize Selenium WebDriver
    print("Initializing Chrome WebDriver...")
    options = webdriver.ChromeOptions()
    # Uncomment the following line to run in headless mode (no browser window)
    # options.add_argument('--headless')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=options)
    
    # Remove automation flags
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    # List to store all scraped data
    all_product_data = []
    
    def save_to_csv(data, filename='amazon_scraped_products.csv'):
        """Save product data to CSV file"""
        if not data:
            return
        
        # Get all unique keys from all products
        all_keys = set()
        for product in data:
            all_keys.update(product.keys())
        
        # Save directly to file (will overwrite if exists)
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(data)
    
    try:
        # Process each barcode (you can limit this for testing)
        # For testing, you might want to only process first few: barcodes[:5]
        consecutive_errors = 0
        for idx, barcode in enumerate(barcodes, 1):
            print(f"\n[{idx}/{len(barcodes)}] Processing barcode: {barcode}")
            
            try:
                product_data = scrape_amazon_product(barcode, driver)
                
                if product_data:
                    all_product_data.append(product_data)
                    consecutive_errors = 0  # Reset error counter on success
                else:
                    consecutive_errors += 1
            except (ConnectionResetError, ConnectionError, Exception) as e:
                error_msg = str(e)
                if 'Connection' in error_msg or '10054' in error_msg:
                    print(f"  ⚠️  Connection lost, will restart browser...")
                else:
                    print(f"  ⚠️  Error: {error_msg[:60]}")
                consecutive_errors += 1
                
            # Restart browser after 3 consecutive errors
            if consecutive_errors >= 3:
                print("\n  ⚠️  Multiple errors detected, restarting browser...")
                try:
                    driver.quit()
                except:
                    pass
                time.sleep(2)
                driver = webdriver.Chrome(options=options)
                consecutive_errors = 0
                print("  ✓ Browser restarted")
            
            # Save to CSV every 5 products
            if len(all_product_data) > 0 and len(all_product_data) % 5 == 0:
                output_file = 'amazon_scraped_products_excel.csv'
                print(f"\n💾 Auto-saving: {len(all_product_data)} products scraped so far...")
                save_to_csv(all_product_data, output_file)
                print(f"✓ Progress saved")
            
            # Add a delay between requests to avoid being blocked (MAXIMUM SPEED)
            time.sleep(0.3)
        
        # Save final results to CSV
        if all_product_data:
            output_file = 'amazon_scraped_products_excel.csv'
            print(f"\n💾 Final save: {len(all_product_data)} products to {output_file}...")
            save_to_csv(all_product_data, output_file)
            print(f"✓ Successfully saved data to {output_file}")
            
            # Clean the data and save cleaned version
            print("\n🧹 Cleaning data...")
            try:
                df = pd.read_csv(output_file, encoding='utf-8-sig')
                df_cleaned = clean_amazon_dataframe(df)
                cleaned_output = 'amazon_cleaned.csv'
                df_cleaned.to_csv(cleaned_output, index=False, encoding='utf-8-sig')
                print(f"✓ Cleaned data saved to {cleaned_output}")
                print(f"  {len(df_cleaned)} rows × {len(df_cleaned.columns)} columns")
            except Exception as e:
                print(f"⚠ Warning: Could not clean data - {e}")
        else:
            print("\n⚠ No product data was collected")
    
    finally:
        # Close the browser
        print("\nClosing browser...")
        driver.quit()
        print("Done!")

if __name__ == "__main__":
    main()
