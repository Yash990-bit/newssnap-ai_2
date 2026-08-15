"""NewsSnap AI - Image Extraction and Validation Utilities."""

import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Minimum dimensions considered a "real" lead image (not an icon or logo)
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200

# Common patterns for URLs that are clearly not article images
_SKIP_URL_PATTERNS = re.compile(
    r"(logo|icon|avatar|spinner|placeholder|pixel|blank|spacer|1x1|tracking)",
    re.IGNORECASE,
)


def extract_og_image(html_str: str, base_url: str = "") -> Optional[str]:
    """
    Extract the Open Graph image URL from an HTML document.

    Args:
        html_str: Raw HTML content of the page.
        base_url: Base URL of the page (used to resolve relative paths).

    Returns:
        Absolute image URL, or None if not found.
    """
    if not html_str:
        return None

    soup = BeautifulSoup(html_str, "html.parser")

    # Try og:image first
    og_tag = soup.find("meta", property="og:image")
    if og_tag and og_tag.get("content"):
        return _resolve_url(og_tag["content"].strip(), base_url)

    # Fallback: twitter:image
    tw_tag = soup.find("meta", attrs={"name": "twitter:image"})
    if tw_tag and tw_tag.get("content"):
        return _resolve_url(tw_tag["content"].strip(), base_url)

    return None


def extract_first_large_image(html_str: str, base_url: str = "") -> Optional[str]:
    """
    Find the first <img> tag that looks like a meaningful article image.

    Skips images with URLs matching common noise patterns (logos, icons, etc.).

    Args:
        html_str: Raw HTML content.
        base_url: Base URL for resolving relative src values.

    Returns:
        Absolute image URL, or None if not found.
    """
    if not html_str:
        return None

    soup = BeautifulSoup(html_str, "html.parser")

    for img in soup.find_all("img", src=True):
        src: str = img["src"].strip()
        if not src or src.startswith("data:"):
            continue
        if _SKIP_URL_PATTERNS.search(src):
            continue

        # Try width/height attributes as a quick filter
        width = _parse_int(img.get("width", ""))
        height = _parse_int(img.get("height", ""))
        if width and width < MIN_IMAGE_WIDTH:
            continue
        if height and height < MIN_IMAGE_HEIGHT:
            continue

        resolved = _resolve_url(src, base_url)
        if resolved:
            return resolved

    return None


def pick_best_image(
    html_str: str,
    base_url: str = "",
    fallback_url: Optional[str] = None,
) -> Optional[str]:
    """
    Return the best available image URL for an article.

    Priority order:
      1. og:image / twitter:image meta tag
      2. First large <img> in the article body
      3. fallback_url (e.g., image already scraped from RSS)

    Args:
        html_str: Raw HTML of the article page.
        base_url: Used to resolve relative URLs.
        fallback_url: Image URL already found by a prior step (e.g., RSS enclosure).

    Returns:
        Best image URL, or None if none found.
    """
    url = extract_og_image(html_str, base_url)
    if url:
        return url

    url = extract_first_large_image(html_str, base_url)
    if url:
        return url

    return fallback_url


def is_valid_image_url(url: Optional[str]) -> bool:
    """
    Return True if *url* looks like a real, usable image URL.

    Checks:
    - Must be a non-empty string
    - Must have http/https scheme
    - Must not match common noise patterns
    """
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if _SKIP_URL_PATTERNS.search(url):
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_url(src: str, base_url: str) -> Optional[str]:
    """Convert a possibly-relative URL to an absolute one."""
    if not src:
        return None
    if src.startswith(("http://", "https://")):
        return src
    if base_url:
        try:
            return urljoin(base_url, src)
        except Exception:
            pass
    return None


def _parse_int(value: str) -> Optional[int]:
    """Parse a string to int, returning None on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
