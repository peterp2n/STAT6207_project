import re
from bs4 import BeautifulSoup
import polars as pl
from pathlib import Path
from datetime import datetime


class Extractor:
    EMPTY_PRODUCT = {
        'isbn': None, 'title': None, 'price': None,
        'rating': None, 'number_of_reviews': None, 'availability': None,
        'isbn_10': None, 'isbn_13': None,
        'publisher': None, 'publication_date': None, 'language': None,
        'length': None, 'width': None, 'height': None,
        'item_weight': None, 'print_length': None,
        'reading_age': None, 'edition': None, 'author': None, 'asin': None,
        'series_name': None, 'best_sellers_rank': None, 'customer_reviews': None,
        'description': None, 'product_url': None, 'book_format': None,
        'scrape_status': None, 'error_message': None
    }

    FIELD_MAPPINGS = {
        'ISBN-10': 'isbn_10', 'ISBN-13': 'isbn_13', 'Publisher': 'publisher',
        'Publication date': 'publication_date', 'Language': 'language',
        'Dimensions': 'dimensions',
        'Item Weight': 'item_weight', 'Item weight': 'item_weight',
        'Print length': 'print_length',
        'Reading age': 'reading_age', 'Edition': 'edition', 'ASIN': 'asin',
        'Series': 'series_name', 'Part of': 'series_name',
        'Part of series': 'series_name', 'Best Sellers Rank': 'best_sellers_rank'
    }

    EUR_TO_USD_RATE = 1.08

    def __init__(self):
        self.results = []
        self.df = None

    @staticmethod
    def safe_convert(value, target_type=float):
        if value is None:
            return None
        cleaned = str(value).replace(',', '').strip()
        if not cleaned:
            return None
        try:
            return target_type(cleaned)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def clean_isbn(isbn_str):
        return ''.join(c for c in (isbn_str or '') if c.isdigit() or c.upper() == 'X') or None

    @staticmethod
    def extract_title(soup):
        title_elem = soup.find('span', id='productTitle')
        return title_elem.text.strip() if title_elem else None

    @staticmethod
    def extract_author(soup):
        byline = soup.find('div', id='bylineInfo')
        if byline:
            raw = ', '.join([a.text.strip() for a in byline.find_all('span', class_='author') if a.text.strip()])
        else:
            raw = ', '.join([c.text.strip() for c in soup.find_all('a', class_='contributor') if c.text.strip()])

        if not raw:
            return None

        cleaned = re.sub(r'\s*\([^)]*\)', '', raw)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned.endswith(','):
            cleaned = cleaned[:-1].strip()

        junk = {'unknown author', 'n/a', 'na', 'various', ''}
        if cleaned.lower() in junk:
            return None

        return cleaned or None

    @staticmethod
    def extract_rating(soup):
        rating_span = soup.find('span', id='acrPopover')
        if rating_span and 'title' in rating_span.attrs:
            raw = rating_span['title'].split()[0]
        else:
            alt_span = soup.find('span', class_='a-icon-alt')
            raw = alt_span.text.split()[0] if alt_span else None

        if raw == "Previous" or not raw:
            return None

        return Extractor.safe_convert(raw, float)

    @staticmethod
    def extract_number_of_reviews(soup):
        rev_span = soup.find('span', id='acrCustomerReviewText')
        if not rev_span:
            return None
        raw = rev_span.text.strip().split()[0]
        return Extractor.safe_convert(raw, int)

    @staticmethod
    def extract_availability(soup):
        avail_div = soup.find('div', id='availability')
        return avail_div.text.strip() if avail_div else None

    @staticmethod
    def extract_description(soup):
        desc_div = soup.find('div', id='bookDescription_feature_div')
        if not desc_div:
            return None
        noscript = desc_div.find('noscript')
        return (noscript.text if noscript else desc_div.text).strip() or None

    @staticmethod
    def extract_product_details(soup):
        details = {}
        bullet_div = soup.find('div', id='detailBullets_feature_div')
        if bullet_div:
            for li in bullet_div.find_all('li'):
                span = li.find('span', class_='a-list-item')
                if span:
                    bold = span.find('span', class_='a-text-bold')
                    if bold:
                        key = re.sub(r'[\s\u200e\u200f]+', ' ', bold.text).strip().rstrip(' :')
                        value = re.sub(r'[\s\u200e\u200f]+', ' ', span.text.replace(bold.text, '')).strip()
                        if value:
                            details[key] = value
        else:
            table = soup.find('table', id='productDetails_techSpec_section_1')
            if table:
                for row in table.find_all('tr'):
                    th = row.find('th')
                    td = row.find('td')
                    if th and td:
                        details[th.text.strip()] = td.text.strip()
        return details

    @staticmethod
    def parse_dimensions(product):
        dim_str = product.pop('dimensions', None)
        if not dim_str:
            return product
        dims = re.findall(r'([\d.]+)', dim_str)
        if len(dims) >= 3:
            float_dims = sorted([float(d) for d in dims], reverse=True)
            product['length'] = float_dims[0]
            product['width'] = float_dims[1]
            product['height'] = float_dims[2]
        return product

    @staticmethod
    def parse_publisher(product):
        pub_str = product.get('publisher')
        if not pub_str:
            return product
        match = re.match(r'^(.*?);\s*(.*?)\s*\((.*?)\)$', pub_str)
        if match:
            product['publisher'] = match.group(1)
            product['edition'] = match.group(2)
            product['publication_date'] = match.group(3)
            return product
        match = re.match(r'^(.*)\s*\((.*)\)$', pub_str)
        if match:
            product['publisher'] = match.group(1)
            product['publication_date'] = match.group(2)
        return product

    @staticmethod
    def extract_product_url(soup):
        canonical = soup.find('link', rel='canonical')
        return canonical.get('href') if canonical else None

    @staticmethod
    def parse_item_weight(product):
        weight_str = product.get('item_weight')
        if not weight_str:
            product['item_weight'] = None
            return product

        match = re.match(r'([\d.]+)\s*(ounces?|oz|grams?|g|kilograms?|kg|pounds?|lb)', weight_str, re.I)
        if not match:
            product['item_weight'] = None  # ← explicitly set to None on failure
            return product

        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith('g'):
            value *= 0.035274
        elif unit.startswith('kg'):
            value *= 35.274
        elif unit.startswith('lb') or unit.startswith('pound'):
            value *= 16

        product['item_weight'] = round(value, 2)  # always float
        return product

    @staticmethod
    def parse_best_sellers_rank(product):
        rank_str = product.get('best_sellers_rank')
        if not rank_str:
            product['best_sellers_rank'] = ""
            return product
        match = re.search(r'#([\d,]+) in Books', rank_str)
        if match:
            product['best_sellers_rank'] = match.group(1).replace(',', '')
        else:
            product['best_sellers_rank'] = ""
        return product

    @staticmethod
    def extract_selected_format_and_price(soup):
        selected = soup.find('div', class_=re.compile(r'swatchElement.*selected'))
        if selected:
            title_span = selected.find('span', class_='slot-title')
            format_str = None
            if title_span:
                text = title_span.get_text(strip=True)
                format_str = re.sub(r'\s*Format:.*$', '', text, flags=re.I).strip()

            price = None
            price_span = selected.find('span', class_='slot-price')
            if price_span:
                aria = price_span.find('span', attrs={'aria-label': True})
                raw = aria['aria-label'] if aria and aria.get('aria-label') else price_span.get_text(strip=True)
                raw = raw.strip()
                is_eur = '€' in raw or 'EUR' in raw.upper()
                num_match = re.search(r'[\d,]+(?:\.\d+)?', raw)
                if num_match:
                    try:
                        value = float(num_match.group(0).replace(',', ''))
                        if is_eur:
                            value *= Extractor.EUR_TO_USD_RATE
                        price = round(value, 2)
                    except ValueError:
                        price = None

            return format_str, price

        subtitle = soup.find('span', id='productSubtitle')
        if subtitle:
            text = subtitle.get_text(strip=True)
            if re.fullmatch(r'\d+(st|nd|rd|th)\s+Edition.*', text, re.I):
                return None, None
            match = re.match(r'([^,]+)', text)
            if match:
                fmt = match.group(1).strip()
                fmt = re.sub(r'\s*–\s*$', '', fmt)
                return fmt or None, None

        return None, None

    def parse(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        product = self.EMPTY_PRODUCT.copy()
        error_messages = []

        extraction_funcs = {
            'title': self.extract_title,
            'author': self.extract_author,
            'rating': self.extract_rating,
            'number_of_reviews': self.extract_number_of_reviews,
            'availability': self.extract_availability,
            'description': self.extract_description,
        }

        for key, func in extraction_funcs.items():
            try:
                product[key] = func(soup)
            except Exception as e:
                error_messages.append(f"Error extracting {key}: {str(e)}")

        book_format, price = self.extract_selected_format_and_price(soup)
        product['book_format'] = book_format
        product['price'] = price

        try:
            details = self.extract_product_details(soup)
            for label, value in details.items():
                mapped_key = self.FIELD_MAPPINGS.get(label)
                if mapped_key:
                    if mapped_key == 'publication_date':
                        try:
                            dt = datetime.strptime(value, "%B %d, %Y")
                            value = dt.strftime("%Y-%m-%d")
                        except ValueError:
                            print(f"Failed to convert publication date: {value}")
                    if mapped_key == 'print_length':
                        match = re.search(r'([\d,]+)', value)
                        if match:
                            value = self.safe_convert(match.group(1).replace(',', ''), int)
                    if mapped_key == 'reading_age':
                        nums = re.findall(r'\d+', value)
                        if nums:
                            lower = float(nums[0])
                            mean_age = (lower + float(nums[1])) / 2 if len(nums) >= 2 else lower
                            value = int(mean_age) if mean_age.is_integer() else mean_age
                        else:
                            value = None
                    product[mapped_key] = value
        except Exception as e:
            error_messages.append(f"Error extracting product details: {str(e)}")

        try:
            product = self.parse_dimensions(product)
        except Exception as e:
            error_messages.append(f"Error parsing dimensions: {str(e)}")

        try:
            product = self.parse_publisher(product)
        except Exception as e:
            error_messages.append(f"Error parsing publisher: {str(e)}")

        try:
            product = self.parse_item_weight(product)
        except Exception as e:
            error_messages.append(f"Error parsing item weight: {str(e)}")

        try:
            product = self.parse_best_sellers_rank(product)
        except Exception as e:
            error_messages.append(f"Error parsing best sellers rank: {str(e)}")

        try:
            isbn_10 = product.get('isbn_10')
            if isbn_10:
                product['isbn_10'] = self.clean_isbn(isbn_10)
        except Exception as e:
            error_messages.append(f"Error cleaning ISBN-10: {str(e)}")

        try:
            isbn_13 = product.get('isbn_13')
            if isbn_13:
                product['isbn_13'] = self.clean_isbn(isbn_13)
                product['isbn'] = product['isbn_13']
        except Exception as e:
            error_messages.append(f"Error cleaning ISBN-13: {str(e)}")

        try:
            asin = product.get('asin')
            product['product_url'] = self.extract_product_url(soup) or (
                f"https://www.amazon.com/dp/{asin}" if asin else None)
        except Exception as e:
            error_messages.append(f"Error setting product URL: {str(e)}")

        try:
            rating = product.get('rating')
            num_reviews = product.get('number_of_reviews')
            if rating and num_reviews:
                product['customer_reviews'] = rating
        except Exception as e:
            error_messages.append(f"Error setting customer reviews: {str(e)}")

        product['scrape_status'] = 'success'
        if error_messages:
            product['scrape_status'] = 'fail'
            product['error_message'] = '; '.join(error_messages)
        elif (not product["isbn"] or not product["title"]):
            product['scrape_status'] = 'fail'

        self.results.append(product)

    def to_dataframe(self):
        self.df = pl.DataFrame(self.results or [self.EMPTY_PRODUCT])
        return self.df

    @staticmethod
    def read_html(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None


if __name__ == "__main__":
    ext = Extractor()
    html_folder = Path("data")

    # All HTML files recursively
    all_paths = list(html_folder.rglob("*.html"))

    # Keep only files where the filename (without .html) is exactly 13 characters
    # OR starts with "978" (your original logic)
    html_paths = [
        p for p in all_paths
        if len(p.stem) == 13 or p.stem.startswith("product_978")
    ]

    print(f"Processing {len(html_paths)} files...")

    for html_path in html_paths:
        content = Extractor.read_html(html_path)
        if content:
            ext.parse(content)

    df = ext.to_dataframe()
    df.write_csv(html_folder / "amazon_cleaned.csv")

    api = (pl.scan_csv(html_folder / "books_api_cleaned.csv", schema_overrides={"isbn": pl.Utf8})
           .select(["page_count", "isbn"]).collect())
    merged2 = df.join(api, left_on='isbn_13', right_on='isbn', how='left', suffix='_api').with_columns(
        # Use original print_length, if present; else use page_count from API
        print_length=
        pl.when(pl.col("print_length").is_not_null()).then(pl.col("print_length"))
        .when(pl.col("page_count").is_not_null() & (pl.col("page_count") != 0)).then(pl.col("page_count"))
        .otherwise(None)
    # Drop the page_count column after merging
    ).drop("page_count")

    # merged2.write_csv(html_folder / "merged.csv")
    print(df)