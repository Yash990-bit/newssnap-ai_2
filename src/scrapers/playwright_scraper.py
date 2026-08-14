"""NewsSnap AI - Playwright Web Scraper."""

import asyncio
import logging
import urllib.robotparser
from datetime import datetime, timezone
from typing import List
from urllib.parse import urljoin, urlparse

from src.scrapers.article_parser import ScrapedArticle
from src.scrapers.browser_pool import browser_pool
from src.scrapers.source_registry import SourceConfig, SourceRegistry
from src.utils.text_utils import clean_html, extract_text, normalize_whitespace

logger = logging.getLogger(__name__)


class PlaywrightScraper:
    """Scraper that uses Playwright to render JS-heavy websites."""

    def __init__(self, config_dir: str | None = None):
        self.registry = SourceRegistry(config_dir=config_dir)
        self.registry.load_from_disk()
        self._robot_parsers = {}

    async def _check_robots_txt(self, url: str) -> bool:
        """Checks if the URL is allowed to be scraped according to robots.txt."""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        if base_url not in self._robot_parsers:
            robots_url = urljoin(base_url, "/robots.txt")
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                # We do this synchronously, it's a small text file fetch
                # For robust async, we might want to use httpx
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get(robots_url, timeout=10.0)
                    rp.parse(response.text.splitlines())
            except Exception as e:
                logger.warning(f"Failed to fetch robots.txt for {base_url}: {e}")
                # If robots.txt can't be fetched, we proceed with caution
                rp.allow_all = True

            self._robot_parsers[base_url] = rp

        rp = self._robot_parsers.get(base_url)
        if hasattr(rp, "allow_all") and rp.allow_all:
            return True
        return rp.can_fetch("*", url)

    async def _fetch_with_retry(self, page, url: str, retries: int = 3) -> bool:
        """Navigates to a URL with exponential backoff retries."""
        backoff_times = [1, 2, 4]
        for attempt in range(retries):
            try:
                # wait_until="domcontentloaded" is usually enough to get the core JS started
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if response and response.status < 400:
                    return True
                else:
                    status = response.status if response else "Unknown"
                    logger.warning(f"Attempt {attempt + 1}: Bad response {status} for {url}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to fetch {url}: {e}")

            if attempt < retries - 1:
                delay = backoff_times[attempt] if attempt < len(backoff_times) else backoff_times[-1]
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        logger.error(f"Failed to fetch {url} after {retries} attempts.")
        return False

    async def scrape_source(self, slug: str) -> List[ScrapedArticle]:
        """Scrapes a source and returns a list of ScrapedArticle objects."""
        config: SourceConfig = self.registry.get_config(slug)
        if not config:
            logger.error(f"No configuration found for source: {slug}")
            return []

        logger.info(f"Starting scrape for {config.name}")

        # 1. Check robots.txt for the article list URL
        if not await self._check_robots_txt(str(config.article_list_url)):
            logger.error(f"Scraping {config.article_list_url} is disallowed by robots.txt")
            return []

        articles: List[ScrapedArticle] = []

        async with browser_pool.get_page() as page:
            success = await self._fetch_with_retry(page, str(config.article_list_url))
            if not success:
                return []

            # Wait a bit for JS to render the links
            await asyncio.sleep(2)

            # 2. Extract article links
            links = []
            if config.article_link_selector:
                # Find all elements matching the selector
                elements = await page.locator(config.article_link_selector).all()
                for el in elements:
                    href = await el.get_attribute("href")
                    if href:
                        links.append(urljoin(str(config.base_url), href))

            # Remove duplicates and limit to a reasonable number for testing
            links = list(set(links))[:20]
            logger.info(f"Found {len(links)} article links for {config.name}")

            # 3. Scrape each article
            for link in links:
                if not await self._check_robots_txt(link):
                    logger.warning(f"Scraping {link} disallowed by robots.txt. Skipping.")
                    continue

                # Rate limiting
                await asyncio.sleep(config.rate_limit_seconds)

                success = await self._fetch_with_retry(page, link)
                if not success:
                    continue

                try:
                    # Extract title
                    title = ""
                    if config.title_selector:
                        title_el = page.locator(config.title_selector).first
                        if await title_el.count() > 0:
                            title = await title_el.inner_text()

                    if not title:
                        title = await page.title()

                    # Extract body
                    body_html = ""
                    if config.body_selector:
                        body_el = page.locator(config.body_selector).first
                        if await body_el.count() > 0:
                            body_html = await body_el.inner_html()

                    # Clean and normalize body text
                    cleaned_html = clean_html(body_html)
                    body_text = extract_text(cleaned_html)
                    body_text = normalize_whitespace(body_text)

                    # Extract image
                    image_url = None
                    if config.image_selector:
                        img_el = page.locator(config.image_selector).first
                        if await img_el.count() > 0:
                            src = await img_el.get_attribute("src")
                            if src:
                                image_url = urljoin(str(config.base_url), src)

                    # Determine category (fallback to "general")
                    category = "general"
                    for keyword, mapped_category in config.category_mapping.items():
                        if keyword.lower() in link.lower():
                            category = mapped_category
                            break

                    # Create ScrapedArticle
                    if title and body_text:
                        article = ScrapedArticle(
                            title=title.strip(),
                            body=body_text,
                            source_url=link,
                            publish_time=datetime.now(timezone.utc),
                            category=category,
                            image_url=image_url,
                        )
                        articles.append(article)
                        logger.info(f"Successfully scraped: {title[:50]}...")
                    else:
                        logger.warning(f"Skipping {link} due to missing title or body.")

                except Exception as e:
                    logger.error(f"Error extracting data from {link}: {e}")

        return articles
