import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from sqlalchemy.orm import Session

from src.config.settings import Category, Language, ScrapeType
from src.models.source import Source

logger = logging.getLogger(__name__)


class SourceConfig(BaseModel):
    """Schema for validating source JSON configurations."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Display name of the source")
    base_url: HttpUrl = Field(..., description="Base URL of the publication")
    article_list_url: HttpUrl = Field(..., description="URL to scrape for article links")

    article_link_selector: Optional[str] = Field(None, description="CSS/XPath selector for article links")
    title_selector: Optional[str] = Field(None, description="CSS/XPath selector for article title")
    body_selector: Optional[str] = Field(None, description="CSS/XPath selector for article content")
    image_selector: Optional[str] = Field(None, description="CSS/XPath selector for main image")

    category_mapping: Dict[str, Category] = Field(
        default_factory=dict, description="Mapping of source categories to internal categories"
    )

    language: Language = Field(default=Language.ENGLISH, description="Primary language of the source")
    scrape_type: ScrapeType = Field(default=ScrapeType.PLAYWRIGHT, description="Method to scrape the source")
    rate_limit_seconds: int = Field(default=2, ge=0, description="Delay between requests to this source")


class SourceRegistry:
    """Registry for managing and syncing news source configurations."""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Default to the source_configs directory relative to this file
            base_dir = Path(__file__).parent
            self.config_dir = base_dir / "source_configs"
        else:
            self.config_dir = Path(config_dir)

        self.configs: Dict[str, SourceConfig] = {}

    def load_from_disk(self) -> int:
        """Load all valid JSON configurations from the config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory {self.config_dir} does not exist. Creating it.")
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return 0

        loaded_count = 0
        self.configs.clear()

        for file_path in self.config_dir.glob("*.json"):
            slug = file_path.stem
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                config = SourceConfig(**data)
                self.configs[slug] = config
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load config {file_path.name}: {str(e)}")

        return loaded_count

    def get_config(self, slug: str) -> Optional[SourceConfig]:
        """Get a specific loaded configuration by its slug."""
        if not self.configs:
            self.load_from_disk()
        return self.configs.get(slug)

    def sync_with_db(self, db: Session) -> None:
        """Sync loaded JSON configurations into the database."""
        if not self.configs:
            self.load_from_disk()

        for slug, config in self.configs.items():
            db_source = db.query(Source).filter(Source.slug == slug).first()

            if not db_source:
                # Create new source
                db_source = Source(
                    slug=slug,
                    name=config.name,
                    base_url=str(config.base_url),
                    article_list_url=str(config.article_list_url),
                    language=config.language,
                    scrape_type=config.scrape_type,
                    rate_limit_seconds=config.rate_limit_seconds,
                    is_active=True,
                )
                db.add(db_source)
                logger.info(f"Created new source in DB: {slug}")
            else:
                # Update existing source
                db_source.name = config.name
                db_source.base_url = str(config.base_url)
                db_source.article_list_url = str(config.article_list_url)
                db_source.language = config.language
                db_source.scrape_type = config.scrape_type
                db_source.rate_limit_seconds = config.rate_limit_seconds
                logger.info(f"Updated existing source in DB: {slug}")

        try:
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to sync sources to DB: {str(e)}")
            raise

    def get_all_sources(self, db: Optional[Session] = None) -> List[Source]:
        """
        Get all sources.
        If db is provided, gets from DB, otherwise returns mock objects from loaded configs.
        """
        if db:
            # Important: Ensure DB is synced before returning? Let's just query
            return db.query(Source).all()
        else:
            # Fallback when no DB session is provided (e.g. CLI test)
            if not self.configs:
                self.load_from_disk()

            mock_sources = []
            for slug, config in self.configs.items():
                source = Source(
                    slug=slug,
                    name=config.name,
                    base_url=str(config.base_url),
                    article_list_url=str(config.article_list_url),
                    language=config.language,
                    scrape_type=config.scrape_type,
                    rate_limit_seconds=config.rate_limit_seconds,
                    is_active=True,
                )
                mock_sources.append(source)
            return mock_sources

    def update_health(self, db: Session, slug: str, success: bool, error_increment: int = 1) -> Optional[Source]:
        """Update health metrics for a given source."""
        source = db.query(Source).filter(Source.slug == slug).first()
        if not source:
            logger.error(f"Cannot update health: Source {slug} not found in DB")
            return None

        source.last_scrape_time = datetime.utcnow()
        if success:
            source.success_count += 1
            # Reset error count on success? Or just leave it?
            # Typically you might want to mark it healthy if it succeeds.
            source.is_healthy = True
        else:
            source.error_count += error_increment
            # If errors exceed threshold, mark unhealthy
            if source.error_count > 5:  # Arbitrary threshold
                source.is_healthy = False

        try:
            db.commit()
            db.refresh(source)
            return source
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update health for {slug}: {str(e)}")
            return None
