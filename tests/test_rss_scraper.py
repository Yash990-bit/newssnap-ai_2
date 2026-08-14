from unittest.mock import AsyncMock, MagicMock, patch

import feedparser
import pytest
from src.config.settings import ScrapeType
from src.scrapers.rss_scraper import RSSScraper
from src.scrapers.source_registry import SourceConfig, SourceRegistry


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=SourceRegistry)
    config = SourceConfig(
        name="Test RSS Source",
        base_url="https://testrss.com",
        article_list_url="https://testrss.com/feed",
        scrape_type=ScrapeType.RSS,
    )
    registry.get_config.return_value = config
    return registry


@pytest.fixture
def rss_scraper(mock_registry):
    return RSSScraper(registry=mock_registry)


@pytest.mark.asyncio
async def test_rss_scraper_full_content(rss_scraper):
    # Mock feedparser response for a full content feed
    mock_feed = feedparser.FeedParserDict()
    mock_feed["bozo"] = 0

    entry1 = feedparser.FeedParserDict()
    entry1["title"] = "Test Article 1"
    entry1["link"] = "https://testrss.com/article1"
    entry1["summary"] = (
        "This is a very long summary that exceeds our 500 character threshold so it won't trigger the static scraper fallback. "
        * 10
    )
    entry1["published_parsed"] = (2026, 8, 14, 12, 0, 0, 4, 226, 0)

    mock_feed["entries"] = [entry1]

    with patch("feedparser.parse", return_value=mock_feed):
        articles = await rss_scraper.scrape_source("testrss")

        assert len(articles) == 1
        assert articles[0].title == "Test Article 1"
        assert articles[0].source_url == "https://testrss.com/article1"
        assert articles[0].body == entry1["summary"]
        assert articles[0].publish_time.year == 2026


@pytest.mark.asyncio
async def test_rss_scraper_link_only_fallback(rss_scraper):
    # Mock feedparser response for a link-only feed (short summary)
    mock_feed = feedparser.FeedParserDict()
    mock_feed["bozo"] = 0

    entry1 = feedparser.FeedParserDict()
    entry1["title"] = "Short Summary Article"
    entry1["link"] = "https://testrss.com/article2"
    entry1["summary"] = "Short summary."
    entry1["published_parsed"] = (2026, 8, 14, 12, 0, 0, 4, 226, 0)

    mock_feed["entries"] = [entry1]

    # Mock the static scraper to return a full body
    rss_scraper.static_scraper.fetch_article = AsyncMock(
        return_value={"body": "This is the full fetched article body.", "image_url": "https://testrss.com/image.jpg"}
    )

    with patch("feedparser.parse", return_value=mock_feed):
        articles = await rss_scraper.scrape_source("testrss")

        assert len(articles) == 1
        assert articles[0].title == "Short Summary Article"
        assert articles[0].body == "This is the full fetched article body."
        assert articles[0].image_url == "https://testrss.com/image.jpg"


@pytest.mark.asyncio
async def test_rss_scraper_invalid_source(rss_scraper):
    rss_scraper.registry.get_config.return_value = None
    articles = await rss_scraper.scrape_source("nonexistent")
    assert len(articles) == 0


@pytest.mark.asyncio
async def test_rss_scraper_wrong_scrape_type(rss_scraper):
    wrong_config = SourceConfig(
        name="Test Playwright",
        base_url="https://test.com",
        article_list_url="https://test.com/feed",
        scrape_type=ScrapeType.PLAYWRIGHT,
    )
    rss_scraper.registry.get_config.return_value = wrong_config

    articles = await rss_scraper.scrape_source("test")
    assert len(articles) == 0
