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
        'part_of_series': None, 'best_sellers_rank': None, 'customer_reviews': None, 'description': None, 'product_url': None,
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

    trans_table = str.maketrans('', '', ''.join(INVISIBLE_CHARS))

    def __init__(self):
        self.results = []
        self.df = None

    @staticmethod
    def clean_single(text):
        if text is None:
            return None
        text = text.translate(extractor.trans_table)
        return ' '.join(text.split())

    @staticmethod
    def clean_multi(text):
        if text is None:
            return None
        text = text.translate(extractor.trans_table)
        return text.strip()

    @staticmethod
    def clean_isbn(isbn_str):
        return ''.join(c for c in (isbn_str or '') if c.isdigit() or c.upper() == 'X') or None

    @staticmethod
    def extract_title(soup):
        title_elem = soup.find('span', id='productTitle')
        if title_elem:
            return extractor.clean_single(title_elem.text)
        return None

    @staticmethod
    def extract_author(soup):
        byline = soup.find('div', id='bylineInfo')
        if byline:
            authors = [extractor.clean_single(a.text) for a in byline.find_all('span', class_='author') if a.text.strip()]
            if authors:
                return ', '.join(authors)
        contribs = [extractor.clean_single(c.text) for c in soup.find_all('a', class_='contributor') if c.text.strip()]
        if contribs:
            return ', '.join(contribs)
        return None

    @staticmethod
    def extract_price(soup):
        price_span = soup.find('span', class_='a-offscreen')
        if price_span:
            return extractor.clean_single(price_span.text)
        return None

    @staticmethod
    def extract_rating(soup):
        rating_span = soup.find('span', id='acrPopover')
        if rating_span and 'title' in rating_span.attrs:
            return extractor.clean_single(rating_span['title']).split()[0]
        alt_span = soup.find('span', class_='a-icon-alt')
        if alt_span:
            return extractor.clean_single(alt_span.text).split()[0]
        return None

    @staticmethod
    def extract_number_of_reviews(soup):
        rev_span = soup.find('span', id='acrCustomerReviewText')
        if rev_span:
            rev_text = extractor.clean_single(rev_span.text).split()[0].replace(',', '')
            return rev_text
        return None

    @staticmethod
    def extract_availability(soup):
        avail_div = soup.find('div', id='availability')
        if avail_div:
            return extractor.clean_single(avail_div.text)
        return None

    @staticmethod
    def extract_features(soup):
        feature_div = soup.find('div', id='feature-bullets')
        if not feature_div:
            return None
        ul = feature_div.find('ul')
        if not ul:
            return None
        features = [
            extractor.clean_single(span.text)
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
        text = noscript.text if noscript else desc_div.text
        return extractor.clean_multi(text) or None

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
                        bold_text = bold.text
                        bold_clean = extractor.clean_single(bold_text)
                        key = bold_clean.rstrip(' :')
                        span_text = span.text
                        span_clean = extractor.clean_single(span_text)
                        value = span_clean.replace(bold_clean, '', 1).strip()
                        if value:
                            details[key] = value
        else:
            table = soup.find('table', id='productDetails_techSpec_section_1')
            if table:
                for row in table.find_all('tr'):
                    th = row.find('th')
                    td = row.find('td')
                    if th and td:
                        key = extractor.clean_single(th.text)
                        value = extractor.clean_single(td.text)
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
            product['width'] = dims[0]
            product['length'] = dims[1]
            product['height'] = dims[2]
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
                            print(f"Failed to convert: {value}")
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
                product['customer_reviews'] = f"{rating} out of 5 stars ({num_reviews} reviews)"
        except Exception as e:
            error_messages.append(f"Error setting customer reviews: {str(e)}")

        if error_messages:
            product['scrape_status'] = 'partial_success' if any(product.values()) else 'failure'
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
            with open(file_path, "r") as reader:
                return reader.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

if __name__ == "__main__":
    ext = Extractor()
    html_folder = Path("data")
    html_path = html_folder / "product_9780064450836.html"
    html_content = Extractor.read_html(html_path)  # Replace with your actual file path
    if html_content:
        ext.parse(html_content)
        df = ext.to_dataframe()
        print(df)

    pass