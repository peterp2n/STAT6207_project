"""Amazon challenge detection and bypass functionality"""

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random


class ChallengeMixin:
    """Handles Amazon challenge detection and bypass"""

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
                    time.sleep(random.uniform(1.0, 2.5))

                    try:
                        button.click()
                    except:
                        self.driver.execute_script("arguments[0].click();", button)

                    print(f"[{query}] ✓ Button clicked, waiting for redirect...")
                    time.sleep(random.uniform(3, 5))
                    return True

                except:
                    continue

            print(f"[{query}] ✗ Could not find 'Continue shopping' button")
            return False

        except Exception as e:
            print(f"[{query}] Error handling continue button: {e}")
            return False
