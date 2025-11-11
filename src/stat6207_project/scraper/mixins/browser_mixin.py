"""Browser setup and lifecycle management"""

from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
import time
import random


class BrowserMixin:
    """Handles browser creation, configuration, and lifecycle"""

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

    def cleanup(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("\n✓ Browser closed")
