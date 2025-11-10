from dataclasses import dataclass, field
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


@dataclass
class Data:
    args: dict
    queries: list = field(init=False)
    driver: webdriver.Firefox = field(init=False)

    def __post_init__(self):
        self.queries = self.args.get("queries")

    def create_torbrowser_webdriver_instance(self) -> bool:
        firefox_options = Options()
        if self.args.get("headless", False):
            firefox_options.add_argument("--headless")

        # Add user agent to avoid detection
        firefox_options.set_preference("general.useragent.override",
                                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0")

        # Specify the path to geckodriver
        service = Service(executable_path=Path("driver/geckodriver"))

        binary = Path("/Applications/Tor Browser.app/Contents/MacOS/firefox")
        if not binary.exists():
            raise ValueError("The binary path to Tor firefox does not exist.")

        firefox_options.binary_location = str(binary)
        firefox_capabilities = webdriver.DesiredCapabilities.FIREFOX
        firefox_capabilities["proxy"] = {
            "proxyType": "MANUAL",
            "socksProxy": "127.0.0.1:9150",
            "socksVersion": 5
        }
        try:
            self.driver = webdriver.Firefox(service=service, options=firefox_options)

            # Maximize window to ensure elements are visible
            self.driver.maximize_window()

            # Wait for Tor Browser to fully load
            time.sleep(5)

            is_success = True
        except Exception as e:
            print(f"Error creating driver: {e}")
            is_success = False

        return is_success

    def scrape(self):
        # Check if connect button exists without waiting long
        try:
            connect_button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, "connectButton"))
            )
            # If found, wait for it to be clickable
            connect_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "connectButton"))
            )
            connect_button.click()

            # Wait for Tor connection
            time.sleep(15)  # Increased wait time for Tor connection
            print("Tor connection established")

        except Exception as e:
            print(f"Connect button not found - assuming Tor already connected: {e}")

        # Navigate to Amazon
        print("Navigating to Amazon...")
        self.driver.get("https://www.amazon.com/ref=nav_logo")

        # Add longer wait for page to load
        time.sleep(5)

        # Debug: Print current URL and title
        print(f"Current URL: {self.driver.current_url}")
        print(f"Page title: {self.driver.title}")

        # Check if we got a CAPTCHA or error page
        page_source = self.driver.page_source.lower()
        if "captcha" in page_source or "robot" in page_source:
            print("WARNING: CAPTCHA or bot detection page detected!")
            # Save screenshot for debugging
            self.driver.save_screenshot("captcha_page.png")
            print("Screenshot saved as captcha_page.png")

        # Try to find the search box with multiple selectors
        wait = WebDriverWait(self.driver, 30)
        search_box = None

        selectors = [
            (By.ID, "twotabsearchtextbox"),
            (By.NAME, "field-keywords"),
            (By.CSS_SELECTOR, "input[type='text'][name='field-keywords']"),
            (By.CSS_SELECTOR, "#nav-search-bar-form input[type='text']")
        ]

        for selector_type, selector_value in selectors:
            try:
                print(f"Trying selector: {selector_type} = {selector_value}")
                search_box = wait.until(
                    EC.element_to_be_clickable((selector_type, selector_value))
                )
                print(f"Found search box with: {selector_type} = {selector_value}")
                break
            except Exception as e:
                print(f"Selector {selector_value} failed: {e}")
                continue

        if not search_box:
            # Save page source for debugging
            with open("page_source.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print("Page source saved to page_source.html")
            raise Exception("Could not find search box with any selector")

        # Clear and enter search query
        search_box.clear()
        search_box.send_keys(self.queries[0])
        search_box.send_keys(Keys.RETURN)

        print("Search submitted, waiting for results...")
        time.sleep(3)

        # Wait for search results
        wait = WebDriverWait(self.driver, 20)
        try:
            results_div = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot"))
            )
            print("Search results loaded successfully")
        except Exception as e:
            print(f"Could not find results: {e}")
            # Save screenshot
            self.driver.save_screenshot("results_error.png")
            print("Screenshot saved as results_error.png")
            raise


def main():
    arguments = {
        "headless": False,
        "queries": ["9780064450836"]
    }
    scraper = Data(arguments)
    status = scraper.create_torbrowser_webdriver_instance()

    if not status:
        print("Failed to create driver")
        return

    try:
        scraper.scrape()
        print("Scraping completed successfully!")
    except Exception as e:
        print(f"Error during scraping: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(scraper, 'driver') and scraper.driver:
            input("Press Enter to close browser...")  # Keep browser open for inspection
            scraper.driver.quit()


if __name__ == "__main__":
    main()
