"""NewsSnap AI - Playwright Browser Pool Management."""

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from playwright.async_api import Browser, Page, Playwright, async_playwright
from playwright_stealth import stealth

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]


class BrowserPool:
    """Manages Playwright browser instances and provides isolated contexts/pages."""

    def __init__(self):
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Starts the playwright and browser instance."""
        async with self._lock:
            if self._playwright is None:
                self._playwright = await async_playwright().start()
                # Use chromium by default, but can be configured
                self._browser = await self._playwright.chromium.launch(headless=True)
                logger.info("Browser pool initialized.")

    async def stop(self) -> None:
        """Stops the browser and playwright instance."""
        async with self._lock:
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            logger.info("Browser pool stopped.")

    def get_random_user_agent(self) -> str:
        """Returns a random user agent from the pool."""
        return random.choice(USER_AGENTS)

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator[Page, None]:
        """
        Yields a new page in an isolated context with a random User-Agent.
        Closes the context when done.
        """
        if not self._browser:
            await self.start()

        # We can't guarantee self._browser isn't None due to types, but start() sets it.
        browser = self._browser
        if not browser:
            raise RuntimeError("Browser failed to start.")

        user_agent = self.get_random_user_agent()
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()
        await stealth(page)
        try:
            yield page
        finally:
            await page.close()
            await context.close()


# Global shared instance
browser_pool = BrowserPool()
