from dataclasses import dataclass
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Optional, Dict
import time
import re


@dataclass
class TorScraper:
    queries: List[str]
    headless: bool = False
    driver: Optional[webdriver.Firefox] = None

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 200) -> str:
        """
        Sanitize a string to make it safe for use as a filename.
        Removes or replaces invalid characters for Windows, Mac, and Linux.
        """
        # Replace invalid characters with underscore
        # Invalid: / \ : * ? " < > | and control characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)

        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)

        # Remove leading/trailing spaces and dots (Windows doesn't allow)
        sanitized = sanitized.strip(' .')

        # If filename is empty after sanitization, use a default
        if not sanitized:
            sanitized = "unnamed"

        # Limit length to avoid filesystem issues
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def create_driver(self) -> bool:
        """Create a single Firefox/Tor browser instance"""
        firefox_options = Options()
        if self.headless:
            firefox_options.add_argument("--headless")

        firefox_options.set_preference("general.useragent.override",
                                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0")

        service = Service(executable_path=Path("driver/geckodriver"))
        binary = Path("/Applications/Tor Browser.app/Contents/MacOS/firefox")

        if not binary.exists():
            raise ValueError("Tor Browser binary not found")

        firefox_options.binary_location = str(binary)
        firefox_options.set_preference("network.proxy.type", 1)
        firefox_options.set_preference("network.proxy.socks", "127.0.0.1")
        firefox_options.set_preference("network.proxy.socks_port", 9150)
        firefox_options.set_preference("network.proxy.socks_version", 5)

        try:
            self.driver = webdriver.Firefox(service=service, options=firefox_options)
            self.driver.maximize_window()
            time.sleep(5)
            print("✓ Browser created successfully")
            return True
        except Exception as e:
            print(f"✗ Error creating driver: {e}")
            return False

    def handle_tor_connection(self, query: str):
        """Handle Tor browser connection screen"""
        try:
            connect_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "connectButton"))
            )
            connect_button.click()
            time.sleep(15)
            print(f"[{query}] Tor connected")
        except:
            print(f"[{query}] Tor already connected")

    def search_amazon(self, query: str) -> bool:
        """Navigate to Amazon and perform search"""
        print(f"[{query}] Navigating to Amazon...")
        self.driver.get("https://www.amazon.com/ref=nav_logo")
        time.sleep(5)

        print(f"[{query}] Current URL: {self.driver.current_url}")

        # Check for CAPTCHA
        if "captcha" in self.driver.page_source.lower():
            print(f"[{query}] ⚠️  CAPTCHA detected!")
            sanitized_query = self.sanitize_filename(query)
            self.driver.save_screenshot(f"captcha_{sanitized_query}.png")
            return False

        # Find search box
        wait = WebDriverWait(self.driver, 30)
        selectors = [
            (By.ID, "twotabsearchtextbox"),
            (By.NAME, "field-keywords"),
        ]

        for selector_type, selector_value in selectors:
            try:
                search_box = wait.until(
                    EC.element_to_be_clickable((selector_type, selector_value))
                )
                search_box.clear()
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                print(f"[{query}] Search submitted")
                time.sleep(5)
                return True
            except:
                continue

        print(f"[{query}] ✗ Search box not found")
        return False

    def click_first_result(self, query: str) -> Optional[Dict]:
        """Click first search result and get product page"""
        wait = WebDriverWait(self.driver, 30)

        try:
            # Wait for results container
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot")))
            print(f"[{query}] Results loaded, looking for first product link...")

            # Scroll to ensure element is in view
            self.driver.execute_script("window.scrollTo(0, 300);")
            time.sleep(2)

            # Try multiple selectors for the first result link
            selectors = [
                (By.CSS_SELECTOR, "div[data-cy='title-recipe'] a.s-link-style"),
                (By.CSS_SELECTOR, "h2.a-size-medium a"),
                (By.CSS_SELECTOR, "div.s-main-slot div[data-component-type='s-search-result'] h2 a"),
                (By.CSS_SELECTOR, "div.s-main-slot a[href*='/dp/']"),
            ]

            first_result = None
            product_url = None

            for selector_type, selector_value in selectors:
                try:
                    print(f"[{query}] Trying selector: {selector_value}")
                    elements = self.driver.find_elements(selector_type, selector_value)

                    # Filter to get actual product links
                    for element in elements:
                        href = element.get_attribute("href")
                        if href and "/dp/" in href and "offer-listing" not in href:
                            if element.is_displayed():
                                first_result = element
                                product_url = href
                                print(f"[{query}] Found clickable product link: {product_url}")
                                break

                    if first_result:
                        break

                except Exception as e:
                    print(f"[{query}] Selector {selector_value} failed: {e}")
                    continue

            if not first_result:
                print(f"[{query}] ✗ Could not find any clickable product link")
                sanitized_query = self.sanitize_filename(query)
                self.driver.save_screenshot(f"no_result_{sanitized_query}.png")
                return {
                    "query": query,
                    "url": None,
                    "html": None,
                    "title": None,
                    "status": "no_result_found"
                }

            # Scroll element into view and click
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", first_result)
            time.sleep(1)

            # Try regular click first
            try:
                print(f"[{query}] Clicking first result...")
                first_result.click()
            except Exception as e:
                print(f"[{query}] Regular click failed, trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", first_result)

            time.sleep(5)

            # Verify we're on a product page
            current_url = self.driver.current_url
            if "/dp/" not in current_url:
                print(f"[{query}] ⚠️  Warning: May not be on product page. URL: {current_url}")

            # Get product page data
            return {
                "query": query,
                "url": current_url,
                "html": self.driver.page_source,
                "title": self.driver.title,
                "status": "success"
            }

        except Exception as e:
            print(f"[{query}] ✗ Error: {e}")
            sanitized_query = self.sanitize_filename(query)
            self.driver.save_screenshot(f"error_{sanitized_query}.png")

            # Save page source for debugging
            with open(Path("data") / f"page_source_{sanitized_query}.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print(f"[{query}] Page source saved for debugging")

            return {
                "query": query,
                "url": None,
                "html": None,
                "title": None,
                "status": "error"
            }

    def scrape_one(self, query: str) -> Dict:
        """Scrape a single query"""
        start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{query}] Starting...")
        print(f"{'=' * 60}")

        try:
            # Search Amazon
            if not self.search_amazon(query):
                return {"query": query, "status": "search_error", "url": None, "html": None}

            # Click first result and get data
            result = self.click_first_result(query)

            elapsed = time.time() - start_time
            print(f"[{query}] ✓ Completed in {elapsed:.1f}s")

            return result

        except Exception as e:
            print(f"[{query}] ✗ Fatal error: {e}")
            return {"query": query, "status": "fatal_error", "url": None, "html": None}

    def scrape_all(self) -> List[Dict]:
        """Scrape all queries sequentially"""
        results = []

        for i, query in enumerate(self.queries, 1):
            print(f"\n📋 Processing query {i}/{len(self.queries)}: {query}")
            result = self.scrape_one(query)
            results.append(result)

            # Small pause between queries
            if i < len(self.queries):
                print(f"\nWaiting 3 seconds before next query...")
                time.sleep(3)

        return results

    def cleanup(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("\n✓ Browser closed")


def main():
    # List of queries to scrape
    queries = [
        "9780064450836",
        "python programming book",
        "web/scraping: guide?",  # Test sanitization with invalid characters
    ]

    scraper = TorScraper(queries=queries, headless=False)

    try:
        # Create browser
        if not scraper.create_driver():
            print("Failed to create browser")
            return

        # Handle Tor connection (only once at the start)
        scraper.handle_tor_connection("initial")

        # Scrape all queries sequentially
        start = time.time()
        results = scraper.scrape_all()
        elapsed = time.time() - start

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"COMPLETED ALL TASKS in {elapsed:.1f}s")
        print(f"{'=' * 60}\n")

        successful = 0
        for result in results:
            query = result['query']
            status = result['status']

            if status == "success":
                successful += 1
                print(f"✓ {query}")
                print(f"  URL: {result['url']}")
                print(f"  Title: {result['title']}")
                print(f"  HTML: {len(result['html'])} bytes")

                # Sanitize query for filename
                sanitized_query = scraper.sanitize_filename(query)
                filename = Path("data") / f"product_{sanitized_query}.html"

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result['html'])
                print(f"  Saved: {filename}\n")
            else:
                print(f"✗ {query} - Status: {status}\n")

        print(f"\nSuccess rate: {successful}/{len(results)}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    main()
