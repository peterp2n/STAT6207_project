"""Data storage functionality"""

from typing import Dict, Optional


class StorageMixin:
    """Handles saving scraped data to disk"""

    def save_result(self, result: Dict) -> bool:
        """Save a single scrape result immediately after scraping

        Success: Saves combined HTML to json_folder/success/product_{safe_query}/
        Failure: Creates empty folder at json_folder/fail/product_{safe_query}/
        """
        try:
            query = result['query']
            status = result['status']
            safe_query = query.replace(' ', '_')[:30]

            if status == "success":
                print(f"✓ {query}")
                print(f"  URL: {result['url']}")
                print(f"  Title: {result['title']}")

                # Combine both HTMLs
                combined_html = self._combine_html(result['html'], result.get('modal_html'))
                print(f"  Combined HTML: {len(combined_html)} bytes")

                # Save combined HTML to success folder
                folder = self.args.get("json_folder") / "success" / f"product_{safe_query}"
                folder.mkdir(parents=True, exist_ok=True)

                filename = folder / f"product_{safe_query}.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(combined_html)
                print(f"  Saved: {filename}\n")
                return True

            else:
                # Create empty fail folder but don't save HTML
                folder = self.args.get("json_folder") / "fail" / f"product_{safe_query}"
                folder.mkdir(parents=True, exist_ok=True)

                print(f"✗ {query} - Status: {status}")
                print(f"  Empty folder created: {folder}\n")
                return False

        except Exception as e:
            print(f"✗ Error saving result for {result.get('query', 'unknown')}: {e}")
            return False

    def _combine_html(self, main_html: str, modal_html: Optional[str]) -> str:
        """Combine main product page HTML with formats modal HTML

        If modal HTML exists, append it with a clear separator comment.
        Otherwise, return just the main HTML.
        """
        if not main_html:
            return ""

        if modal_html:
            separator = "\n\n<!-- ========== FORMATS MODAL HTML BELOW ========== -->\n\n"
            return main_html + separator + modal_html
        else:
            return main_html
