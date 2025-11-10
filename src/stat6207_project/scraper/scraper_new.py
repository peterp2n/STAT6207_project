from dataclasses import field
from pathlib import Path
from src.stat6207_project.scraper.Data import Data
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Optional

class Scraper(Data):

    driver = field(default=None, init=False)
    queries: list[str] = field(default_factory=list)

    def __init__(self, args: dict):
        super().__init__(args)
        chrome_options = Options()
        if args.get("headless", False):
            chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.queries = field(default_factory=list)

    def scrape(self):
        self.driver.get("https://www.amazon.com/ref=nav_logo")
        wait = WebDriverWait(self.driver, 5)

        search_box = wait.until(EC.presence_of_element_located((By.ID, "twotabsearchtextbox")))
        search_query = "9780064450836"
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        wait = WebDriverWait(self.driver, 10)
        results_div = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot"))
        )
        self.driver.close()

def main():

    arguments = {
        "db_path": Path("data") / "Topic1_dataset.sqlite",
        "headless": True,

    }
    scraper = Scraper(arguments)
    scraper.load_all_tables()

    tables = scraper.table_holder
    scraper.scrape()

if __name__ == "__main__":
    main()
    pass