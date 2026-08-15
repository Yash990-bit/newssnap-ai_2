import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import feedparser

from src.scrapers.article_parser import ScrapedArticle
from src.scrapers.source_registry import SourceConfig, SourceRegistry
from src.scrapers.static_scraper import StaticScraper

logger = logging.getLogger(__name__)


class RSSScraper:
    """Scraper for extracting articles from RSS/Atom feeds."""

    def __init__(self, registry: Optional[SourceRegistry] = None):
        self.registry = registry or SourceRegistry()
        self.static_scraper = StaticScraper()

    async def scrape_source(self, source_slug: str) -> List[ScrapedArticle]:
        """Scrape articles from a configured RSS/Atom source."""
        config = self.registry.get_config(source_slug)
        if not config:
            logger.error(f"No configuration found for source: {source_slug}")
            return []

        if config.scrape_type.value != "rss":
            logger.error(f"Source {source_slug} is not configured for RSS scraping (type: {config.scrape_type}).")
            return []

        logger.info(f"Starting RSS scrape for {config.name} at {config.article_list_url}")

        try:
            # Directly parse the feed URL using feedparser. This avoids external HTTP requests during tests.
            feed = feedparser.parse(str(config.article_list_url))
        except Exception as e:
            logger.error(f"Failed to parse RSS feed for {source_slug}: {str(e)}")
            return []

        # Parse the feed data (already obtained above)
        # feed variable already contains parsed feed
        # If using the httpx path, feed_data would be raw text, but now feed is direct.
        # Ensure feed variable is defined.
        # No further action needed here.

        if feed.bozo and feed.bozo_exception:
            logger.warning(
                f"Feedparser reported a bozo exception (malformed XML) for {source_slug}: {feed.bozo_exception}"
            )

        articles = []
        for entry in feed.entries:
            try:
                article = await self._parse_entry(entry, config)
                if article:
                    articles.append(article)
            except Exception:
                logger.exception(f"Error parsing RSS entry for {source_slug}")

        logger.info(f"Finished RSS scrape for {source_slug}. Extracted {len(articles)} articles.")
        return articles

    async def _parse_entry(self, entry: feedparser.FeedParserDict, config: SourceConfig) -> Optional[ScrapedArticle]:
        """Parse a single RSS/Atom entry into a ScrapedArticle."""
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()

        if not title or not link:
            logger.warning("Skipping entry missing title or link.")
            return None

        # Parse date
        publish_time = datetime.now(timezone.utc)
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                publish_time = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
            except Exception as e:
                logger.warning(f"Failed to parse date for entry {title}: {e}")

        # Extract summary/content
        content = ""
        if hasattr(entry, "content"):
            content = entry.content[0].value
        elif hasattr(entry, "summary"):
            content = entry.summary
        elif hasattr(entry, "description"):
            content = entry.description

        # Image extraction
        image_url = None
        if hasattr(entry, "media_content") and len(entry.media_content) > 0:
            image_url = entry.media_content[0].get("url")
        elif hasattr(entry, "media_thumbnail") and len(entry.media_thumbnail) > 0:
            image_url = entry.media_thumbnail[0].get("url")
        elif hasattr(entry, "links"):
            # check for enclosures
            for link_item in entry.links:
                if link_item.get("rel") == "enclosure" and link_item.get("type", "").startswith("image/"):
                    image_url = link_item.get("href")
                    break

        # Author extraction
        author = entry.get("author") or entry.get("creator")

        # Check if full content or link-only
        body = content
        # Arbitrary threshold: if the content is less than 500 characters or missing, treat as link-only
        if len(content) < 500:
            logger.debug(f"Content short ({len(content)} chars), treating as link-only for {link}")
            static_data = await self.static_scraper.fetch_article(link)
            if static_data.get("body"):
                body = static_data["body"]

            # Use static image if none found in RSS
            if not image_url and static_data.get("image_url"):
                image_url = static_data["image_url"]

        if not body:
            logger.warning(f"Skipping entry due to empty body: {title}")
            return None

        # Determine category using mapping (simple matching, full category detection is for Issue 7)
        # For now, just assign a default or try to match tags
        category = "national"  # fallback
        if hasattr(entry, "tags"):
            for tag in entry.tags:
                tag_term = tag.get("term", "").lower()
                if tag_term in config.category_mapping:
                    category = config.category_mapping[tag_term].value
                    break

        return ScrapedArticle(
            title=title,
            body=body,
            source_url=link,
            publish_time=publish_time.replace(
                tzinfo=None
            ),  # Database typically stores naive or handles timezone via settings
            category=category,
            image_url=image_url,
            author=author,
        )
