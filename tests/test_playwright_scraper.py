from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.scrapers.playwright_scraper import PlaywrightScraper
from src.scrapers.source_registry import SourceConfig


@pytest.fixture
def mock_config():
    return SourceConfig(
        name="Test Source",
        base_url="https://test.com",
        article_list_url="https://test.com/latest",
        article_link_selector=".link",
        title_selector=".title",
        body_selector=".body",
        image_selector=".image",
        category_mapping={"tech": "technology"},
        language="en",
        scrape_type="playwright",
        rate_limit_seconds=1,
    )


@pytest.mark.asyncio
async def test_fetch_with_retry_success():
    scraper = PlaywrightScraper()
    mock_page = AsyncMock()

    # First attempt fails, second succeeds
    mock_response_fail = MagicMock()
    mock_response_fail.status = 500
    mock_response_success = MagicMock()
    mock_response_success.status = 200

    mock_page.goto.side_effect = [mock_response_fail, mock_response_success]

    # Patch asyncio.sleep to not actually sleep during tests
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        success = await scraper._fetch_with_retry(mock_page, "https://test.com")

        assert success is True
        assert mock_page.goto.call_count == 2
        # Backoff on first retry is 1s
        mock_sleep.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_fetch_with_retry_failure():
    scraper = PlaywrightScraper()
    mock_page = AsyncMock()

    # Always fail
    mock_response_fail = MagicMock()
    mock_response_fail.status = 500
    mock_page.goto.return_value = mock_response_fail

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        success = await scraper._fetch_with_retry(mock_page, "https://test.com", retries=3)

        assert success is False
        assert mock_page.goto.call_count == 3
        assert mock_sleep.call_count == 2
        # Backoff delays 1s, then 2s
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)


@pytest.mark.asyncio
@patch("src.scrapers.playwright_scraper.browser_pool.get_page")
async def test_scrape_source(mock_get_page, mock_config):
    # Setup mocks
    scraper = PlaywrightScraper()
    scraper.registry = MagicMock()
    scraper.registry.get_config.return_value = mock_config

    # Mock robots.txt to allow everything
    scraper._check_robots_txt = AsyncMock(return_value=True)

    # Mock page and browser context
    mock_page = AsyncMock()

    # Mock article list links
    mock_link_el = AsyncMock()
    mock_link_el.get_attribute.return_value = "/article/1-tech"
    mock_page.locator.return_value.all.return_value = [mock_link_el]

    # Mock article content parsing
    mock_title_el = AsyncMock()
    mock_title_el.count.return_value = 1
    mock_title_el.inner_text.return_value = "Test Title"

    mock_body_el = AsyncMock()
    mock_body_el.count.return_value = 1
    mock_body_el.inner_html.return_value = "<p>Clean <b>Text</b></p><script>alert(1)</script>"

    mock_img_el = AsyncMock()
    mock_img_el.count.return_value = 1
    mock_img_el.get_attribute.return_value = "/img.jpg"

    def locator_side_effect(selector):
        if selector == mock_config.title_selector:
            mock_locator = MagicMock()
            mock_locator.first = mock_title_el
            return mock_locator
        elif selector == mock_config.body_selector:
            mock_locator = MagicMock()
            mock_locator.first = mock_body_el
            return mock_locator
        elif selector == mock_config.image_selector:
            mock_locator = MagicMock()
            mock_locator.first = mock_img_el
            return mock_locator
        elif selector == mock_config.article_link_selector:
            mock_locator = MagicMock()
            mock_locator.all = AsyncMock(return_value=[mock_link_el])
            return mock_locator
        return MagicMock()

    mock_page.locator = MagicMock(side_effect=locator_side_effect)

    # Mock _fetch_with_retry to immediately return True
    scraper._fetch_with_retry = AsyncMock(return_value=True)

    # Mock context manager for browser_pool.get_page
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_page
    mock_get_page.return_value = mock_context

    # Run the test
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        articles = await scraper.scrape_source("test_slug")

        assert len(articles) == 1
        article = articles[0]
        assert article.title == "Test Title"
        assert article.body == "Clean Text"
        assert article.image_url == "https://test.com/img.jpg"
        assert article.source_url == "https://test.com/article/1-tech"
        assert article.category == "technology"

        # Verify rate limiting sleep was called
        mock_sleep.assert_any_call(mock_config.rate_limit_seconds)
