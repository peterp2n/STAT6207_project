import aiohttp
from aiohttp_socks import ProxyConnector
import asyncio


async def scrape_via_tor():
    connector = ProxyConnector.from_url('socks5://127.0.0.1:9150')

    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get('https://www.amazon.com/s?k=9780064450836') as resp:
            html = await resp.text()
            return html

def main():

    html = asyncio.run(scrape_via_tor())
    print(html[:500])  # Print first 500 characters of the HTML

if __name__ == "__main__":
    main()