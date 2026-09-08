"""NewsSnap AI - Image extraction and processing pipeline (Issue 15).

Provides:
- Lead image extraction with og:image, first-large-image, and favicon fallbacks
- Image validation (rejects images < 200x200)
- Smart 16:9 crop resizing
- JPEG optimizer to keep output < 500 KB
- Category placeholder system for all 19 categories
- In-memory LRU image cache to prevent re-downloading
"""

import io
import logging
import os
import re
from functools import lru_cache
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum accepted image dimensions
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200

# Target 16:9 dimensions for snaps
SNAP_WIDTH = 1080
SNAP_HEIGHT = 607  # 1080 * 9 / 16

# Max output file size in bytes (500 KB)
MAX_FILE_SIZE_BYTES = 500 * 1024

# Timeout for HTTP image downloads
DOWNLOAD_TIMEOUT = 10.0

# Placeholders directory
_PLACEHOLDER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "placeholders",
)

# Noise patterns - reject URLs matching these
_SKIP_URL_PATTERNS = re.compile(
    r"(logo|icon|avatar|spinner|placeholder|pixel|blank|spacer|1x1|tracking|favicon)",
    re.IGNORECASE,
)

# All 19 category slugs (from settings.Category enum)
CATEGORY_SLUGS = [
    "national",
    "international",
    "politics",
    "business",
    "finance",
    "sports",
    "technology",
    "science",
    "automobile",
    "education",
    "health",
    "entertainment",
    "lifestyle",
    "crime",
    "environment",
    "jobs",
    "defence",
    "real_estate",
    "opinion",
]

# Category accent colors used when generating placeholder images
CATEGORY_COLORS = {
    "national": (63, 114, 175),
    "international": (41, 128, 185),
    "politics": (155, 89, 182),
    "business": (39, 174, 96),
    "finance": (22, 160, 133),
    "sports": (230, 126, 34),
    "technology": (52, 73, 94),
    "science": (26, 188, 156),
    "automobile": (192, 57, 43),
    "education": (243, 156, 18),
    "health": (231, 76, 60),
    "entertainment": (142, 68, 173),
    "lifestyle": (241, 196, 15),
    "crime": (44, 62, 80),
    "environment": (39, 174, 96),
    "jobs": (52, 152, 219),
    "defence": (127, 140, 141),
    "real_estate": (211, 84, 0),
    "opinion": (149, 165, 166),
}


# ---------------------------------------------------------------------------
# URL Helpers
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
        return int(str(value).strip().split(".")[0])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Image Extraction
# ---------------------------------------------------------------------------


def extract_og_image(html_str: str, base_url: str = "") -> Optional[str]:
    """Extract og:image or twitter:image meta tag URL from HTML.

    Args:
        html_str: Raw HTML content.
        base_url: Page base URL for resolving relative paths.

    Returns:
        Absolute image URL or None.
    """
    if not html_str:
        return None
    soup = BeautifulSoup(html_str, "html.parser")

    og_tag = soup.find("meta", property="og:image")
    if og_tag and og_tag.get("content"):
        return _resolve_url(og_tag["content"].strip(), base_url)

    tw_tag = soup.find("meta", attrs={"name": "twitter:image"})
    if tw_tag and tw_tag.get("content"):
        return _resolve_url(tw_tag["content"].strip(), base_url)

    return None


def extract_first_large_image(html_str: str, base_url: str = "") -> Optional[str]:
    """Find first <img> in HTML that passes noise and size heuristics.

    Args:
        html_str: Raw HTML content.
        base_url: Page base URL for resolving relative paths.

    Returns:
        Absolute image URL or None.
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


def extract_favicon_url(html_str: str, base_url: str = "") -> Optional[str]:
    """Extract favicon URL as last-resort fallback.

    Args:
        html_str: Raw HTML content.
        base_url: Page base URL.

    Returns:
        Absolute favicon URL or None.
    """
    if not html_str:
        return None
    soup = BeautifulSoup(html_str, "html.parser")

    for rel in ("icon", "shortcut icon", "apple-touch-icon"):
        link = soup.find("link", rel=lambda r: r and rel in r)
        if link and link.get("href"):
            return _resolve_url(link["href"].strip(), base_url)

    # Construct default favicon path
    if base_url:
        parsed = urlparse(base_url)
        return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"

    return None


def pick_best_image_url(
    html_str: str,
    base_url: str = "",
    fallback_url: Optional[str] = None,
) -> Optional[str]:
    """Return best image URL for an article using priority chain.

    Priority:
      1. og:image / twitter:image
      2. First large <img> in body
      3. fallback_url (e.g., RSS enclosure)
      4. favicon

    Args:
        html_str: Raw HTML content.
        base_url: Page URL.
        fallback_url: Pre-fetched fallback (e.g., from RSS).

    Returns:
        Best image URL or None.
    """
    url = extract_og_image(html_str, base_url)
    if url:
        return url

    url = extract_first_large_image(html_str, base_url)
    if url:
        return url

    if fallback_url:
        return fallback_url

    return extract_favicon_url(html_str, base_url)


# ---------------------------------------------------------------------------
# Image Download (with LRU cache)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _download_cached(url: str) -> bytes:
    """Download image bytes with LRU caching to prevent re-downloading.

    Args:
        url: Image URL to download.

    Returns:
        Raw image bytes.

    Raises:
        ValueError: If download fails.
    """
    try:
        response = httpx.get(url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        raise ValueError(f"Failed to download image from {url}: {exc}") from exc


def download_image(url: str) -> Image.Image:
    """Download an image URL and return a PIL Image (RGB).

    Results are cached in memory to avoid duplicate HTTP requests.

    Args:
        url: Image URL.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If download or decode fails.
    """
    raw = _download_cached(url)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as exc:
        raise ValueError(f"Could not decode image from {url}: {exc}") from exc


def clear_image_cache() -> None:
    """Clear the in-memory image download cache."""
    _download_cached.cache_clear()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_image_url(url: Optional[str]) -> bool:
    """Return True if URL has http/https scheme and no noise patterns.

    Args:
        url: Image URL string.

    Returns:
        True if URL looks like a real image.
    """
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if _SKIP_URL_PATTERNS.search(url):
        return False
    return True


def validate_image(img: Image.Image) -> Tuple[bool, str]:
    """Validate a PIL Image meets minimum quality standards.

    Rejects images that are:
    - Smaller than 200x200 pixels
    - Not RGB/RGBA (likely corrupt)

    Args:
        img: PIL Image to validate.

    Returns:
        (is_valid, reason) tuple. reason is empty string when valid.
    """
    w, h = img.size
    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
        return False, f"Image too small: {w}x{h} (min {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT})"
    if img.mode not in ("RGB", "RGBA", "L"):
        return False, f"Unsupported image mode: {img.mode}"
    return True, ""


# ---------------------------------------------------------------------------
# Resizing (16:9 smart crop)
# ---------------------------------------------------------------------------


def resize_to_16x9(
    img: Image.Image,
    target_width: int = SNAP_WIDTH,
    target_height: int = SNAP_HEIGHT,
) -> Image.Image:
    """Resize and center-crop image to 16:9 aspect ratio.

    Strategy:
    1. Scale image so the shorter dimension fills the target.
    2. Center-crop to exact target dimensions.

    Args:
        img: Source PIL Image.
        target_width: Output width in pixels.
        target_height: Output height in pixels.

    Returns:
        Cropped and resized PIL Image.
    """
    src_w, src_h = img.size
    target_ratio = target_width / target_height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider — scale by height, crop width
        scale = target_height / src_h
        new_w = int(src_w * scale)
        new_h = target_height
    else:
        # Source is taller — scale by width, crop height
        scale = target_width / src_w
        new_w = target_width
        new_h = int(src_h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    img = img.crop((left, top, left + target_width, top + target_height))

    return img


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def optimize_image(
    img: Image.Image,
    max_bytes: int = MAX_FILE_SIZE_BYTES,
    start_quality: int = 85,
) -> bytes:
    """Encode image as JPEG under the given byte budget.

    Iteratively reduces JPEG quality until file size is below max_bytes.

    Args:
        img: PIL Image to encode.
        max_bytes: Maximum allowed file size in bytes.
        start_quality: Starting JPEG quality (1-95).

    Returns:
        JPEG bytes within the size budget.
    """
    quality = start_quality
    buf = io.BytesIO()

    while quality >= 10:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            break
        quality -= 5

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Category Placeholder System
# ---------------------------------------------------------------------------


def get_placeholder_path(category: str) -> str:
    """Return the filesystem path for a category's placeholder image.

    Args:
        category: Category slug (e.g., 'sports', 'technology').

    Returns:
        Absolute path to the placeholder PNG file.
    """
    slug = category.lower() if category.lower() in CATEGORY_SLUGS else "national"
    return os.path.join(_PLACEHOLDER_DIR, f"{slug}.png")


def generate_placeholder_image(
    category: str,
    width: int = SNAP_WIDTH,
    height: int = SNAP_HEIGHT,
) -> Image.Image:
    """Generate a solid-color placeholder PIL Image for a category.

    Args:
        category: Category slug.
        width: Placeholder width.
        height: Placeholder height.

    Returns:
        PIL Image filled with the category's accent color.
    """
    color = CATEGORY_COLORS.get(category.lower(), (100, 100, 100))
    img = Image.new("RGB", (width, height), color)
    return img


def get_placeholder_image(category: str) -> Image.Image:
    """Load an existing placeholder PNG or generate a new one.

    Checks the placeholders directory first; falls back to generating
    a solid-color image if the file does not exist.

    Args:
        category: Category slug.

    Returns:
        PIL Image for the category placeholder.
    """
    path = get_placeholder_path(category)
    if os.path.isfile(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return generate_placeholder_image(category)


def ensure_placeholders_exist() -> None:
    """Generate and save placeholder PNGs for all 19 categories.

    Creates `src/assets/placeholders/<category>.png` for any category
    that does not yet have a placeholder file.
    """
    os.makedirs(_PLACEHOLDER_DIR, exist_ok=True)
    for slug in CATEGORY_SLUGS:
        path = os.path.join(_PLACEHOLDER_DIR, f"{slug}.png")
        if not os.path.isfile(path):
            img = generate_placeholder_image(slug)
            img.save(path, format="PNG")
            logger.debug("Created placeholder: %s", path)


# ---------------------------------------------------------------------------
# High-level Pipeline Entry Point
# ---------------------------------------------------------------------------


def process_image(
    url: Optional[str],
    category: str = "national",
    target_width: int = SNAP_WIDTH,
    target_height: int = SNAP_HEIGHT,
) -> Tuple[Image.Image, bool]:
    """Full image pipeline: download, validate, resize, fallback to placeholder.

    Args:
        url: Image URL to process (may be None).
        category: Category slug for placeholder fallback.
        target_width: Target output width.
        target_height: Target output height.

    Returns:
        (PIL Image, used_placeholder) tuple.
        used_placeholder is True when the placeholder was used instead.
    """
    if url and is_valid_image_url(url):
        try:
            img = download_image(url)
            valid, reason = validate_image(img)
            if valid:
                img = resize_to_16x9(img, target_width, target_height)
                return img, False
            else:
                logger.info("Image validation failed (%s): %s", reason, url)
        except Exception as exc:
            logger.warning("Image download/process error for %s: %s", url, exc)

    # Fallback: category placeholder
    placeholder = get_placeholder_image(category)
    placeholder = resize_to_16x9(placeholder, target_width, target_height)
    return placeholder, True
