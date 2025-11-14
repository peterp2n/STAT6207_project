import re
from bs4 import BeautifulSoup
import polars as pl
from pathlib import Path
from datetime import datetime

class Extractor:
    EMPTY_PRODUCT = {
        'isbn': None, 'title': None, 'price': None,
        'rating': None, 'number_of_reviews': None, 'availability': None,
        'features': None, 'isbn_10': None, 'isbn_13': None,
        'publisher': None, 'publication_date': None, 'language': None,
        'length': None, 'width': None, 'height': None,
        'item_weight': None, 'print_length': None,
        'reading_age': None, 'edition': None, 'author': None, 'asin': None,
        'part_of_series': None, 'best_sellers_rank': None, 'customer_reviews': None, 'description': None,
        'product_url': None,
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
        'Part of': 'part_of_series', 'Part of series': 'part_of_series', 'Best Sellers Rank': 'best_sellers_rank'
    }

    EUR_TO_USD_RATE = 1.08

    def __init__(self):
        self.results = []
        self.df = None

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
            authors = [a.text.strip() for a in byline.find_all('span', class_='author') if a.text.strip()]
            if authors:
                return ', '.join(authors)
        contribs = [c.text.strip() for c in soup.find_all('a', class_='contributor') if c.text.strip()]
        if contribs:
            return ', '.join(contribs)
        return None

    @staticmethod
    def extract_price(soup):
        price_span = soup.find('span', class_='a-offscreen')
        return price_span.text.strip() if price_span else None

    @staticmethod
    def extract_rating(soup):
        rating_span = soup.find('span', id='acrPopover')
        if rating_span and 'title' in rating_span.attrs:
            return rating_span['title'].split()[0]
        alt_span = soup.find('span', class_='a-icon-alt')
        return alt_span.text.split()[0] if alt_span else None

    @staticmethod
    def extract_number_of_reviews(soup):
        rev_span = soup.find('span', id='acrCustomerReviewText')
        if rev_span:
            rev_text = rev_span.text.strip().split()[0].replace(',', '')
            return rev_text
        return None

    @staticmethod
    def extract_availability(soup):
        avail_div = soup.find('div', id='availability')
        return avail_div.text.strip() if avail_div else None

    @staticmethod
    def extract_features(soup):
        feature_div = soup.find('div', id='feature-bullets')
        if not feature_div:
            return None
        ul = feature_div.find('ul')
        if not ul:
            return None
        features = [
            span.text.strip()
            for li in ul.find_all('li')
            if (span := li.find('span', class_='a-list-item')) and span.text.strip()
        ]
        return features or None

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
                        bold_text = re.sub(r'[\s\u200e\u200f]+', ' ', bold.text).strip()
                        key = bold_text.rstrip(' :')
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
                        key = th.text.strip()
                        value = td.text.strip()
                        if key and value:
                            details[key] = value
        return details

    @staticmethod
    def parse_dimensions(product):
        dim_str = product.pop('dimensions', None)
        if not dim_str:
            return product
        dims = re.findall(r'([\d.]+)', dim_str)
        if len(dims) >= 3:
            float_dims = sorted([float(d) for d in dims], reverse=True)
            product['length'] = str(float_dims[0])
            product['width'] = str(float_dims[1])
            product['height'] = str(float_dims[2])
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
            return product
        match = re.match(r'([\d.]+)\s*(ounces?|oz|grams?|g|kilograms?|kg|pounds?|lb)', weight_str, re.I)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit.startswith('g'):  # gram or grams
                value *= 0.035274
            elif unit.startswith('kg'):  # kilogram or kilograms
                value *= 35.274
            elif unit.startswith('lb') or unit.startswith('pound'):  # pound or pounds or lb
                value *= 16
            product['item_weight'] = str(round(value, 2))
        return product

    @staticmethod
    def parse_best_sellers_rank(product):
        rank_str = product.get('best_sellers_rank')
        if not rank_str:
            product['best_sellers_rank'] = ""
            return product

        match = re.search(r'#([\d,]+) in Books', rank_str)
        if match:
            clean_rank = match.group(1).replace(',', '')
            product['best_sellers_rank'] = clean_rank
        else:
            product['best_sellers_rank'] = ""
        return product

    @staticmethod
    def parse_price(raw_price):
        if not raw_price:
            return None

        raw_price = raw_price.strip()

        is_eur = '€' in raw_price or 'EUR' in raw_price.upper()
        is_usd = '$' in raw_price

        num_match = re.search(r'[\d,]+(?:\.\d+)?', raw_price)
        if not num_match:
            return None

        clean_num = num_match.group(0).replace(',', '')
        try:
            value = float(clean_num)
        except ValueError:
            return None

        if is_eur:
            value *= Extractor.EUR_TO_USD_RATE

        return f"{value:.2f}"

    def parse(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        product = self.EMPTY_PRODUCT.copy()
        error_messages = []

        extraction_funcs = {
            'title': self.extract_title,
            'author': self.extract_author,
            'price': self.extract_price,
            'rating': self.extract_rating,
            'number_of_reviews': self.extract_number_of_reviews,
            'availability': self.extract_availability,
            'features': self.extract_features,
            'description': self.extract_description,
        }

        for key, func in extraction_funcs.items():
            try:
                product[key] = func(soup)
            except Exception as e:
                error_messages.append(f"Error extracting {key}: {str(e)}")

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
                            value = match.group(1).replace(',', '')
                    if mapped_key == 'reading_age':
                        nums = re.findall(r'\d+', value)
                        if nums:
                            lower = float(nums[0])
                            if len(nums) >= 2:
                                upper = float(nums[1])
                                mean_age = (lower + upper) / 2
                            else:
                                mean_age = lower
                            if mean_age.is_integer():
                                value = str(int(mean_age))
                            else:
                                value = f"{mean_age:.1f}"
                        else:
                            value = ""
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

        # Clean/convert price after extraction
        raw_price = product.get('price')
        if raw_price:
            product['price'] = self.parse_price(raw_price)

        try:
            rating = product.get('rating')
            num_reviews = product.get('number_of_reviews')
            if rating and num_reviews:
                product['customer_reviews'] = rating
        except Exception as e:
            error_messages.append(f"Error setting customer reviews: {str(e)}")

        if error_messages:
            product['scrape_status'] = 'partial_success' if any(v is not None for v in product.values()) else 'failure'
            product['error_message'] = '; '.join(error_messages)
        else:
            product['scrape_status'] = 'success'

        self.results.append(product)

    def to_dataframe(self):
        if not self.results:
            self.df = pl.DataFrame(schema=list(self.EMPTY_PRODUCT.keys()))
        else:
            self.df = pl.DataFrame(self.results)
        return self.df

    @staticmethod
    def read_html(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as reader:
                return reader.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None


if __name__ == "__main__":
    ext = Extractor()
    html_folder = Path("data")
    html_paths = list(html_folder.rglob("*.html"))[:100]  # limit to 100 files for testing

    for html_path in html_paths:
        html_content = Extractor.read_html(html_path)
        if html_content:
            ext.parse(html_content)

    df = ext.to_dataframe()
    print(df)