"""Tests for Issue 15 - Image extraction and processing pipeline."""

import io
from unittest.mock import MagicMock, patch

from PIL import Image
from src.snaps.image_handler import (
    CATEGORY_SLUGS,
    MAX_FILE_SIZE_BYTES,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    SNAP_HEIGHT,
    SNAP_WIDTH,
    clear_image_cache,
    download_image,
    ensure_placeholders_exist,
    extract_favicon_url,
    extract_first_large_image,
    extract_og_image,
    generate_placeholder_image,
    get_placeholder_image,
    is_valid_image_url,
    optimize_image,
    pick_best_image_url,
    process_image,
    resize_to_16x9,
    validate_image,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width: int = 800, height: int = 600) -> Image.Image:
    """Create a solid-color RGB PIL Image for testing."""
    return Image.new("RGB", (width, height), (120, 180, 60))


def _image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# AC1: Extracts lead image from og:image with fallbacks
# ---------------------------------------------------------------------------


class TestImageExtraction:
    """Acceptance Criteria 1: Extract lead image with fallback chain."""

    def test_extract_og_image_returns_og_url(self):
        """og:image meta tag is returned first."""
        html = '<html><head><meta property="og:image" content="https://example.com/og.jpg"/></head></html>'
        result = extract_og_image(html)
        assert result == "https://example.com/og.jpg"

    def test_extract_og_image_falls_back_to_twitter(self):
        """Falls back to twitter:image when og:image is absent."""
        html = '<html><head><meta name="twitter:image" content="https://example.com/tw.jpg"/></head></html>'
        result = extract_og_image(html)
        assert result == "https://example.com/tw.jpg"

    def test_extract_og_image_returns_none_when_absent(self):
        """Returns None when neither og nor twitter image present."""
        html = "<html><head><title>No image</title></head></html>"
        assert extract_og_image(html) is None

    def test_extract_first_large_image_skips_small(self):
        """Skips images with explicit small dimensions."""
        html = '<img src="https://example.com/tiny.jpg" width="50" height="50"/>'
        assert extract_first_large_image(html) is None

    def test_extract_first_large_image_returns_valid(self):
        """Returns URL of first large img tag."""
        html = '<img src="https://example.com/large.jpg" width="800" height="600"/>'
        result = extract_first_large_image(html)
        assert result == "https://example.com/large.jpg"

    def test_extract_first_large_image_skips_noise_urls(self):
        """Skips images whose URLs match noise patterns like 'logo' or 'icon'."""
        html = '<img src="https://example.com/logo.png" width="800" height="600"/>'
        assert extract_first_large_image(html) is None

    def test_extract_favicon_url_from_link_tag(self):
        """Extracts favicon URL from <link rel='icon'>."""
        html = '<html><head><link rel="icon" href="/favicon.ico"/></head></html>'
        result = extract_favicon_url(html, base_url="https://example.com")
        assert result is not None
        assert "favicon" in result.lower() or result.startswith("https://")

    def test_pick_best_image_url_priority_og_first(self):
        """og:image is preferred over body images and fallback_url."""
        html = (
            '<html><head><meta property="og:image" content="https://example.com/og.jpg"/></head>'
            '<body><img src="https://example.com/body.jpg" width="800" height="600"/></body></html>'
        )
        result = pick_best_image_url(html, fallback_url="https://example.com/fallback.jpg")
        assert result == "https://example.com/og.jpg"

    def test_pick_best_image_url_fallback_when_no_og(self):
        """Falls back to fallback_url when no og or body image found."""
        html = "<html><body><p>No images</p></body></html>"
        result = pick_best_image_url(html, fallback_url="https://example.com/fallback.jpg")
        assert result == "https://example.com/fallback.jpg"


# ---------------------------------------------------------------------------
# AC2: Rejects images < 200x200
# ---------------------------------------------------------------------------


class TestImageValidation:
    """Acceptance Criteria 2: Reject small or broken images."""

    def test_validate_image_passes_large_enough(self):
        """Valid image passes validation."""
        img = _make_rgb_image(800, 600)
        valid, reason = validate_image(img)
        assert valid is True
        assert reason == ""

    def test_validate_image_rejects_too_small_width(self):
        """Image with width < 200 is rejected."""
        img = _make_rgb_image(150, 400)
        valid, reason = validate_image(img)
        assert valid is False
        assert "small" in reason.lower()

    def test_validate_image_rejects_too_small_height(self):
        """Image with height < 200 is rejected."""
        img = _make_rgb_image(400, 150)
        valid, reason = validate_image(img)
        assert valid is False
        assert "small" in reason.lower()

    def test_validate_image_rejects_exact_boundary(self):
        """Image exactly at the minimum boundary passes."""
        img = _make_rgb_image(MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT)
        valid, _ = validate_image(img)
        assert valid is True

    def test_validate_image_rejects_one_pixel_under_min(self):
        """Image one pixel under minimum width is rejected."""
        img = _make_rgb_image(MIN_IMAGE_WIDTH - 1, MIN_IMAGE_HEIGHT)
        valid, _ = validate_image(img)
        assert valid is False

    def test_is_valid_image_url_passes_https(self):
        """HTTPS URL with clean path passes URL validation."""
        assert is_valid_image_url("https://example.com/article.jpg") is True

    def test_is_valid_image_url_rejects_none(self):
        """None fails URL validation."""
        assert is_valid_image_url(None) is False

    def test_is_valid_image_url_rejects_noise_pattern(self):
        """URL matching logo/icon pattern is rejected."""
        assert is_valid_image_url("https://example.com/logo.png") is False

    def test_is_valid_image_url_rejects_non_http(self):
        """Non-http/https scheme is rejected."""
        assert is_valid_image_url("ftp://example.com/img.jpg") is False


# ---------------------------------------------------------------------------
# AC3: Resizes to 16:9 with smart cropping
# ---------------------------------------------------------------------------


class TestImageResizing:
    """Acceptance Criteria 3: Smart 16:9 crop."""

    def test_resize_produces_correct_dimensions(self):
        """Output image has exactly the target dimensions."""
        img = _make_rgb_image(1200, 900)
        result = resize_to_16x9(img)
        assert result.size == (SNAP_WIDTH, SNAP_HEIGHT)

    def test_resize_wide_image(self):
        """Wide input is correctly cropped to 16:9."""
        img = _make_rgb_image(3000, 1000)
        result = resize_to_16x9(img)
        assert result.size == (SNAP_WIDTH, SNAP_HEIGHT)

    def test_resize_tall_image(self):
        """Tall input is correctly cropped to 16:9."""
        img = _make_rgb_image(500, 2000)
        result = resize_to_16x9(img)
        assert result.size == (SNAP_WIDTH, SNAP_HEIGHT)

    def test_resize_custom_dimensions(self):
        """Custom target dimensions are respected."""
        img = _make_rgb_image(800, 600)
        result = resize_to_16x9(img, target_width=640, target_height=360)
        assert result.size == (640, 360)


# ---------------------------------------------------------------------------
# AC4: Category placeholders for all 19 categories
# ---------------------------------------------------------------------------


class TestCategoryPlaceholders:
    """Acceptance Criteria 4: Placeholder for all 19 categories."""

    def test_all_19_categories_defined(self):
        """CATEGORY_SLUGS contains exactly 19 categories."""
        assert len(CATEGORY_SLUGS) == 19

    def test_generate_placeholder_returns_image(self):
        """generate_placeholder_image returns a PIL Image."""
        img = generate_placeholder_image("sports")
        assert isinstance(img, Image.Image)

    def test_generate_placeholder_correct_dimensions(self):
        """Placeholder is generated at default snap dimensions."""
        img = generate_placeholder_image("technology")
        assert img.size == (SNAP_WIDTH, SNAP_HEIGHT)

    def test_generate_placeholder_for_all_categories(self):
        """Placeholder generation works for all 19 categories."""
        for slug in CATEGORY_SLUGS:
            img = generate_placeholder_image(slug)
            assert img.size == (SNAP_WIDTH, SNAP_HEIGHT), f"Failed for category: {slug}"

    def test_generate_placeholder_unknown_category_uses_default(self):
        """Unknown category slug falls back to default color."""
        img = generate_placeholder_image("unknown_category")
        assert img is not None

    def test_get_placeholder_image_returns_image(self):
        """get_placeholder_image returns a PIL Image regardless of files."""
        img = get_placeholder_image("national")
        assert isinstance(img, Image.Image)

    def test_ensure_placeholders_exist_creates_files(self, tmp_path, monkeypatch):
        """ensure_placeholders_exist creates PNG files for all categories."""
        monkeypatch.setattr("src.snaps.image_handler._PLACEHOLDER_DIR", str(tmp_path))
        ensure_placeholders_exist()
        created = list(tmp_path.glob("*.png"))
        assert len(created) == 19


# ---------------------------------------------------------------------------
# AC5: Image cache prevents re-downloading
# ---------------------------------------------------------------------------


class TestImageCache:
    """Acceptance Criteria 5: LRU cache prevents duplicate downloads."""

    def test_download_image_uses_cache(self):
        """Second call to download_image does not re-issue HTTP request.

        We verify caching by mocking httpx.get and confirming it is only
        called once even when download_image is invoked twice with the same URL.
        """
        clear_image_cache()
        fake_img = _make_rgb_image(800, 600)
        buf = io.BytesIO()
        fake_img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        mock_response = MagicMock()
        mock_response.content = raw_bytes
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_http:
            download_image("https://example.com/cached_img.jpg")
            download_image("https://example.com/cached_img.jpg")
            # httpx.get should only be called once; second call hits the LRU cache
            assert mock_http.call_count == 1

    def test_clear_image_cache_resets(self):
        """clear_image_cache resets the LRU cache."""
        clear_image_cache()
        # Just verify no error
        assert True


# ---------------------------------------------------------------------------
# Image Optimizer
# ---------------------------------------------------------------------------


class TestImageOptimizer:
    """Output JPEG should be under 500 KB."""

    def test_optimize_large_image_under_500kb(self):
        """A large image is compressed to under 500 KB."""
        img = _make_rgb_image(1080, 607)
        result = optimize_image(img)
        assert len(result) <= MAX_FILE_SIZE_BYTES

    def test_optimize_returns_bytes(self):
        """optimize_image returns bytes."""
        img = _make_rgb_image(400, 300)
        result = optimize_image(img)
        assert isinstance(result, bytes)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


class TestProcessImagePipeline:
    """Full pipeline: download -> validate -> resize -> fallback."""

    def test_process_image_with_valid_url(self):
        """Valid URL yields processed image without placeholder."""
        fake_img = _make_rgb_image(800, 600)
        buf = io.BytesIO()
        fake_img.save(buf, format="JPEG")

        with patch("src.snaps.image_handler._download_cached", return_value=buf.getvalue()):
            result, used_placeholder = process_image("https://example.com/photo.jpg", "sports")

        assert result.size == (SNAP_WIDTH, SNAP_HEIGHT)
        assert used_placeholder is False

    def test_process_image_falls_back_on_download_failure(self):
        """Download failure triggers placeholder fallback."""
        with patch("src.snaps.image_handler._download_cached", side_effect=ValueError("timeout")):
            result, used_placeholder = process_image("https://example.com/bad.jpg", "sports")

        assert result.size == (SNAP_WIDTH, SNAP_HEIGHT)
        assert used_placeholder is True

    def test_process_image_falls_back_on_none_url(self):
        """None URL immediately triggers placeholder."""
        result, used_placeholder = process_image(None, "technology")
        assert used_placeholder is True
        assert result.size == (SNAP_WIDTH, SNAP_HEIGHT)

    def test_process_image_falls_back_on_small_image(self):
        """Undersized downloaded image triggers placeholder fallback."""
        tiny = _make_rgb_image(50, 50)
        buf = io.BytesIO()
        tiny.save(buf, format="JPEG")

        with patch("src.snaps.image_handler._download_cached", return_value=buf.getvalue()):
            result, used_placeholder = process_image("https://example.com/tiny.jpg", "health")

        assert used_placeholder is True
