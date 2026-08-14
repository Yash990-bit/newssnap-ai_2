import asyncio
import logging
from typing import Dict, Optional

import httpx
from bs4 import BeautifulSoup

from src.utils.text_utils import clean_html, extract_text

logger = logging.getLogger(__name__)


class StaticScraper:
    """Scraper for fetching article content statically using httpx and BeautifulSoup."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def fetch_article(self, url: str, max_retries: int = 3) -> Dict[str, Optional[str]]:
        """
        Fetch article content statically.
        Returns a dictionary with 'body' and optionally 'image_url'.
        """
        result = {"body": None, "image_url": None}

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(url, headers=self.headers)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.text, "html.parser")

                    # Try to extract og:image if present
                    og_image = soup.find("meta", property="og:image")
                    if og_image and og_image.get("content"):
                        result["image_url"] = og_image["content"]

                    # Clean the HTML
                    soup = clean_html(response.text)

                    # Extract text
                    text = extract_text(soup)
                    if text and len(text) > 100:
                        result["body"] = text

                    return result

                except httpx.HTTPError as e:
                    logger.warning(f"Static scrape failed for {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(f"Failed to fetch {url} after {max_retries} attempts.")

        return result
