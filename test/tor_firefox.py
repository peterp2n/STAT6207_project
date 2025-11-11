from dataclasses import dataclass, field
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.stat6207_project.scraper.Data import Data
from typing import List, Optional, Dict
from webdriver_manager.firefox import GeckoDriverManager
import time


@dataclass
class TorScraper(Data):
    queries: List[str] = field(default_factory=list, init=False)
    headless: bool = field(default=False, init=False)
    driver: Optional[webdriver.Firefox] = field(default=None, init=False)

    def create_driver(self) -> bool:
        """Create a single Firefox/Tor browser instance"""
        firefox_options = Options()
        if self.headless:
            firefox_options.add_argument("--headless")

        firefox_options.set_preference("general.useragent.override",
                                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0")

        service = Service(GeckoDriverManager().install())
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
            self.driver.save_screenshot(f"captcha_{query.replace(' ', '_')}.png")
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
                time.sleep(5)  # Increased wait for results to load
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
                # Based on your HTML - title link with data-cy attribute
                (By.CSS_SELECTOR, "div[data-cy='title-recipe'] a.s-link-style"),
                # H2 with link
                (By.CSS_SELECTOR, "h2.a-size-medium a"),
                # More general - any product title link
                (By.CSS_SELECTOR, "div.s-main-slot div[data-component-type='s-search-result'] h2 a"),
                # Even more general - first link that goes to /dp/
                (By.CSS_SELECTOR, "div.s-main-slot a[href*='/dp/']"),
            ]

            first_result = None
            product_url = None

            for selector_type, selector_value in selectors:
                try:
                    print(f"[{query}] Trying selector: {selector_value}")
                    elements = self.driver.find_elements(selector_type, selector_value)

                    # Filter to get actual product links (not "More Buying Choices" etc)
                    for element in elements:
                        href = element.get_attribute("href")
                        if href and "/dp/" in href and "offer-listing" not in href:
                            # Make sure element is visible
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
                self.driver.save_screenshot(f"no_result_{query.replace(' ', '_')}.png")
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
                # If regular click fails, use JavaScript click
                print(f"[{query}] Regular click failed, trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", first_result)

            time.sleep(5)  # Wait for product page to load

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
            self.driver.save_screenshot(f"error_{query.replace(' ', '_')}.png")

            # Save page source for debugging
            with open(f"page_source_{query.replace(' ', '_')}.html", "w", encoding="utf-8") as f:
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

    def save_result(self, result: Dict) -> bool:
        """Save a single scrape result immediately after scraping"""
        try:
            query = result['query']
            status = result['status']

            if status == "success":
                print(f"✓ {query}")
                print(f"  URL: {result['url']}")
                print(f"  Title: {result['title']}")
                print(f"  HTML: {len(result['html'])} bytes")

                # Save HTML
                safe_query = query.replace(' ', '_')[:30]
                folder = self.args.get("json_folder") / f"product_{safe_query}"
                folder.mkdir(parents=True, exist_ok=True)

                filename = folder / f"product_{safe_query}.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result['html'])
                print(f"  Saved: {filename}\n")
                return True
            else:
                print(f"✗ {query} - Status: {status}\n")
                return False

        except Exception as e:
            print(f"✗ Error saving result for {result.get('query', 'unknown')}: {e}")
            return False

    def scrape_all(self) -> List[Dict]:
        """Scrape all queries sequentially and save each result immediately"""
        results = []
        successful = 0
        failed = 0
        total = len(self.queries)

        for i, query in enumerate(self.queries, 1):
            print(f"\n📋 Processing query {i}/{total}: {query}")

            # Scrape the query
            result = self.scrape_one(query)
            results.append(result)

            # Save immediately after scraping and update counters
            if self.save_result(result):
                successful += 1
            else:
                failed += 1

            # Print current progress with counts
            print(f"📊 Progress: {i}/{total} | ✓ Success: {successful} | ✗ Failed: {failed}")

            # Small pause between queries
            if i < total:
                print(f"\nWaiting 3 seconds before next query...")
                time.sleep(3)

        # Print final summary
        print(f"\n{'=' * 60}")
        print(f"COMPLETED: {successful} successful, {failed} failed out of {total} total")
        print(f"Success rate: {successful / total * 100:.1f}%")
        print(f"{'=' * 60}\n")

        return results

    def cleanup(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("\n✓ Browser closed")


def main():
    arguments = {
        "db_path": Path("data") / "Topic1_dataset.sqlite",
        "headless": True,
        "json_folder": Path("data") / "scrapes",
    }

    scraper = TorScraper(arguments)
    scraper.load_all_tables()
    scraper.load_queries()

    try:
        # Create browser
        if not scraper.create_driver():
            print("Failed to create browser")
            return

        # Handle Tor connection (only once at the start)
        scraper.handle_tor_connection("initial")

        # Scrape all queries - results are saved incrementally
        start = time.time()
        results = scraper.scrape_all()
        elapsed = time.time() - start

        print(f"\nTotal execution time: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user - partial results have been saved")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    main()
