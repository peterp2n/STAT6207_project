from dataclasses import dataclass, field
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.stat6207_project.data.Data import Data
from typing import List, Optional, Dict
from webdriver_manager.firefox import GeckoDriverManager
import time
import random


@dataclass
class TorScraper(Data):
    queries: List[str] = field(default_factory=list, init=False)
    headless: bool = field(default=False, init=False)
    driver: Optional[webdriver.Firefox] = field(default=None, init=False)

    def create_driver(self) -> bool:
        """Create a single Firefox/Tor browser instance with anti-detection"""
        firefox_options = Options()
        if self.headless:
            firefox_options.add_argument("--headless")

        # Randomize user agent from pool
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:115.0) Gecko/20100101 Firefox/115.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
        ]
        firefox_options.set_preference("general.useragent.override", random.choice(user_agents))

        # Anti-detection preferences
        firefox_options.set_preference("dom.webdriver.enabled", False)
        firefox_options.set_preference("useAutomationExtension", False)

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

    def detect_challenge_type(self) -> str:
        """Detect what type of Amazon challenge page we're on"""
        page_source = self.driver.page_source.lower()

        # Check for "Continue Shopping" button challenge
        if "continue shopping" in page_source and "/errors/validatecaptcha" in page_source:
            return "continue_button"

        # Check for traditional CAPTCHA with image
        if "captcha" in page_source and "captchacharacters" in page_source:
            return "image_captcha"

        # Check for CAPTCHA by URL pattern
        if "api-services-support.amazon.com" in self.driver.current_url:
            return "image_captcha"

        return "none"

    def handle_continue_shopping(self, query: str) -> bool:
        """Automatically click 'Continue shopping' button if present"""
        try:
            # Multiple possible selectors for the button
            button_selectors = [
                (By.CSS_SELECTOR, "button[type='submit'][alt='Continue shopping']"),
                (By.CSS_SELECTOR, "button.a-button-text"),
                (By.XPATH, "//button[contains(text(), 'Continue shopping')]"),
                (By.XPATH, "//form[@action='/errors/validateCaptcha']//button[@type='submit']"),
            ]

            wait = WebDriverWait(self.driver, 5)

            for selector_type, selector_value in button_selectors:
                try:
                    button = wait.until(
                        EC.element_to_be_clickable((selector_type, selector_value))
                    )

                    print(f"[{query}] Found 'Continue shopping' button, clicking...")

                    # Add slight random delay to appear human-like
                    time.sleep(random.uniform(1.0, 2.5))

                    # Try regular click first
                    try:
                        button.click()
                    except:
                        # Fallback to JavaScript click
                        self.driver.execute_script("arguments[0].click();", button)

                    print(f"[{query}] ✓ Button clicked, waiting for redirect...")

                    # Wait for page to load after button click
                    time.sleep(random.uniform(3, 5))

                    return True

                except:
                    continue

            print(f"[{query}] ✗ Could not find 'Continue shopping' button")
            return False

        except Exception as e:
            print(f"[{query}] Error handling continue button: {e}")
            return False

    def search_amazon(self, query: str) -> bool:
        """Navigate to Amazon and perform search with challenge handling"""
        print(f"[{query}] Navigating to Amazon...")
        self.driver.get("https://www.amazon.com/ref=nav_logo")
        time.sleep(random.uniform(4, 6))

        print(f"[{query}] Current URL: {self.driver.current_url}")

        # Detect and handle challenges
        challenge_type = self.detect_challenge_type()

        if challenge_type == "continue_button":
            print(f"[{query}] ⚠️  Continue shopping challenge detected")
            if self.handle_continue_shopping(query):
                # After clicking, check again for any remaining challenges
                time.sleep(2)
                challenge_type = self.detect_challenge_type()
            else:
                self.driver.save_screenshot(f"continue_fail_{query.replace(' ', '_')}.png")
                return False

        if challenge_type == "image_captcha":
            print(f"[{query}] ⚠️  Image CAPTCHA detected!")
            self.driver.save_screenshot(f"captcha_{query.replace(' ', '_')}.png")
            # Optional: Add manual solving here
            # input(f"[{query}] Solve CAPTCHA and press Enter to continue...")
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

                # Human-like typing with random delays
                search_box.clear()
                time.sleep(random.uniform(0.5, 1.0))

                # Type with slight delays between characters (optional but more human-like)
                for char in query:
                    search_box.send_keys(char)
                    time.sleep(random.uniform(0.05, 0.15))

                time.sleep(random.uniform(0.5, 1.0))
                search_box.send_keys(Keys.RETURN)

                print(f"[{query}] Search submitted")
                time.sleep(random.uniform(4, 6))
                return True
            except:
                continue

        print(f"[{query}] ✗ Search box not found")
        return False

    def click_first_result(self, query: str) -> Optional[Dict]:
        """Click first search result and get product page with challenge handling"""
        wait = WebDriverWait(self.driver, 30)

        try:
            # Wait for results container
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot")))
            print(f"[{query}] Results loaded")

            # Check for challenges on search results page
            challenge_type = self.detect_challenge_type()
            if challenge_type == "continue_button":
                print(f"[{query}] Challenge detected on results page, handling...")
                if not self.handle_continue_shopping(query):
                    return {
                        "query": query,
                        "url": None,
                        "html": None,
                        "title": None,
                        "status": "challenge_failed"
                    }
                # Wait after handling challenge
                time.sleep(random.uniform(2, 3))
                # Re-verify results are still loaded
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot")))

            elif challenge_type == "image_captcha":
                print(f"[{query}] ⚠️  Image CAPTCHA on results page")
                self.driver.save_screenshot(f"captcha_results_{query.replace(' ', '_')}.png")
                return {
                    "query": query,
                    "url": None,
                    "html": None,
                    "title": None,
                    "status": "captcha_blocked"
                }

            print(f"[{query}] Looking for first product link...")

            # Scroll to ensure elements are in view
            self.driver.execute_script("window.scrollTo(0, 300);")
            time.sleep(random.uniform(1.5, 2.5))

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
            time.sleep(random.uniform(0.8, 1.5))

            # Try regular click first
            try:
                print(f"[{query}] Clicking first result...")
                first_result.click()
            except Exception as e:
                # If regular click fails, use JavaScript click
                print(f"[{query}] Regular click failed, trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", first_result)

            # Wait for page navigation
            time.sleep(random.uniform(4, 6))

            # Check for challenges on product page
            current_url = self.driver.current_url
            print(f"[{query}] Navigated to: {current_url}")

            challenge_type = self.detect_challenge_type()
            if challenge_type == "continue_button":
                print(f"[{query}] Challenge detected on product page, handling...")
                if not self.handle_continue_shopping(query):
                    return {
                        "query": query,
                        "url": current_url,
                        "html": None,
                        "title": None,
                        "status": "challenge_failed_product_page"
                    }
                # Wait after handling challenge
                time.sleep(random.uniform(2, 4))
                current_url = self.driver.current_url
                print(f"[{query}] After challenge, now at: {current_url}")

            elif challenge_type == "image_captcha":
                print(f"[{query}] ⚠️  Image CAPTCHA on product page")
                self.driver.save_screenshot(f"captcha_product_{query.replace(' ', '_')}.png")
                return {
                    "query": query,
                    "url": current_url,
                    "html": None,
                    "title": None,
                    "status": "captcha_blocked_product_page"
                }

            # Verify we're on a product page
            if "/dp/" not in current_url:
                print(f"[{query}] ⚠️  Warning: May not be on product page. URL: {current_url}")

            # Wait for product page to fully load
            try:
                wait.until(EC.presence_of_element_located((By.ID, "productTitle")))
                print(f"[{query}] ✓ Product page loaded successfully")
            except:
                print(f"[{query}] Warning: Product title not found, but continuing...")

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
            try:
                with open(f"page_source_{query.replace(' ', '_')}.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"[{query}] Page source saved for debugging")
            except:
                pass

            return {
                "query": query,
                "url": self.driver.current_url if self.driver else None,
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

            # Small pause between queries with randomization
            if i < total:
                wait_time = random.uniform(2.5, 4.0)
                print(f"\nWaiting {wait_time:.1f} seconds before next query...")
                time.sleep(wait_time)

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
        "headless": False,  # Set to True for headless mode
        "json_folder": Path("data") / "scrapes",
    }

    scraper = TorScraper(arguments)
    scraper.load_all_tables()
    que = scraper.table_holder.get("products").select("barcode2").unique(
        maintain_order=True).collect().to_series().to_list()
    scraper.load_queries(queries=que[:5])

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
