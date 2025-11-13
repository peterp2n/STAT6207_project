"""Amazon page navigation and interaction"""

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Optional, Dict
import time
import random


class NavigationMixin:
    """Handles Amazon navigation and interaction"""

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
                time.sleep(2)
                challenge_type = self.detect_challenge_type()
            else:
                self.driver.save_screenshot(f"continue_fail_{query.replace(' ', '_')}.png")
                return False

        if challenge_type == "image_captcha":
            print(f"[{query}] ⚠️  Image CAPTCHA detected!")
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
                time.sleep(random.uniform(0.5, 1.0))

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

    def extract_formats_modal(self, query: str) -> Optional[str]:
        """Click 'See all formats and editions' button and extract modal HTML"""
        wait = WebDriverWait(self.driver, 15)

        try:
            print(f"[{query}] Looking for formats button...")

            # Selectors for the "See all formats and editions" button
            button_selectors = [
                (By.ID, "morpheus-ingress-link"),
                (By.CSS_SELECTOR, "a#morpheus-ingress-link"),
                (By.XPATH, "//a[contains(text(), 'See all formats and editions')]"),
            ]

            formats_button = None
            for selector_type, selector_value in button_selectors:
                try:
                    formats_button = wait.until(
                        EC.element_to_be_clickable((selector_type, selector_value))
                    )
                    print(f"[{query}] Found formats button with selector: {selector_value}")
                    break
                except:
                    continue

            if not formats_button:
                print(f"[{query}] ⚠️ Formats button not found, skipping modal extraction")
                return None

            # Scroll button into view and click
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", formats_button)
            time.sleep(random.uniform(0.5, 1.0))

            try:
                formats_button.click()
            except:
                self.driver.execute_script("arguments[0].click();", formats_button)

            print(f"[{query}] Clicked formats button, waiting for modal...")
            time.sleep(random.uniform(2, 3))

            # Wait for modal container to be visible
            modal_selectors = [
                (By.ID, "bfae-desktop-main-content"),
                (By.CSS_SELECTOR, "div#bfae-desktop-main-content"),
                (By.CSS_SELECTOR, "div.sidesheetWidget"),
            ]

            modal_element = None
            for selector_type, selector_value in modal_selectors:
                try:
                    modal_element = wait.until(
                        EC.visibility_of_element_located((selector_type, selector_value))
                    )
                    print(f"[{query}] Modal loaded with selector: {selector_value}")
                    break
                except:
                    continue

            if not modal_element:
                print(f"[{query}] ⚠️ Modal not visible, may not have loaded")
                return None

            # Extract modal HTML
            modal_html = modal_element.get_attribute("outerHTML")
            print(f"[{query}] ✓ Extracted modal HTML: {len(modal_html)} bytes")

            return modal_html

        except Exception as e:
            print(f"[{query}] ⚠️ Error extracting formats modal: {e}")
            return None

    def click_first_result(self, query: str) -> Optional[Dict]:
        """Click first search result and get product page with challenge handling"""
        wait = WebDriverWait(self.driver, 30)

        try:
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
                        "modal_html": None,
                        "title": None,
                        "status": "challenge_failed"
                    }
                time.sleep(random.uniform(2, 3))
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot")))

            elif challenge_type == "image_captcha":
                print(f"[{query}] ⚠️  Image CAPTCHA on results page")
                self.driver.save_screenshot(f"captcha_results_{query.replace(' ', '_')}.png")
                return {
                    "query": query,
                    "url": None,
                    "html": None,
                    "modal_html": None,
                    "title": None,
                    "status": "captcha_blocked"
                }

            print(f"[{query}] Looking for first product link...")

            self.driver.execute_script("window.scrollTo(0, 300);")
            time.sleep(random.uniform(1.5, 2.5))

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
                self.driver.save_screenshot(f"no_result_{query.replace(' ', '_')}.png")
                return {
                    "query": query,
                    "url": None,
                    "html": None,
                    "modal_html": None,
                    "title": None,
                    "status": "no_result_found"
                }

            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", first_result)
            time.sleep(random.uniform(0.8, 1.5))

            try:
                print(f"[{query}] Clicking first result...")
                first_result.click()
            except Exception as e:
                print(f"[{query}] Regular click failed, trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", first_result)

            time.sleep(random.uniform(4, 6))

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
                        "modal_html": None,
                        "title": None,
                        "status": "challenge_failed_product_page"
                    }
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
                    "modal_html": None,
                    "title": None,
                    "status": "captcha_blocked_product_page"
                }

            if "/dp/" not in current_url:
                print(f"[{query}] ⚠️  Warning: May not be on product page. URL: {current_url}")

            try:
                wait.until(EC.presence_of_element_located((By.ID, "productTitle")))
                print(f"[{query}] ✓ Product page loaded successfully")
            except:
                print(f"[{query}] Warning: Product title not found, but continuing...")

            # Get main page HTML
            main_html = self.driver.page_source

            # Extract formats modal HTML
            modal_html = self.extract_formats_modal(query)

            return {
                "query": query,
                "url": current_url,
                "html": main_html,
                "modal_html": modal_html,
                "title": self.driver.title,
                "status": "success"
            }

        except Exception as e:
            print(f"[{query}] ✗ Error: {e}")
            self.driver.save_screenshot(f"error_{query.replace(' ', '_')}.png")

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
                "modal_html": None,
                "title": None,
                "status": "error"
            }
