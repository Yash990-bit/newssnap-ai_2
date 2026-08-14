"""NewsSnap AI - Article Parser and Models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ScrapedArticle:
    """Represents an article extracted from a scraper before database insertion."""

    title: str
    body: str
    source_url: str
    publish_time: datetime
    category: str
    image_url: Optional[str] = None
    author: Optional[str] = None
