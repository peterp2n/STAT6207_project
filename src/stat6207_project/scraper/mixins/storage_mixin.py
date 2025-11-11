"""Data storage functionality"""

from typing import Dict


class StorageMixin:
    """Handles saving scraped data to disk"""

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
