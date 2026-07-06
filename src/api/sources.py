import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.scrapers.source_registry import SourceConfig, SourceRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["sources"])
registry = SourceRegistry()


@router.on_event("startup")
def startup_event():
    """Load configs on app startup."""
    count = registry.load_from_disk()
    logger.info(f"Loaded {count} source configurations on startup.")


@router.get("")
def list_sources(db: Session = Depends(get_db)):
    """List all sources (syncs with DB if needed)."""
    # Ensure they are synced at least once or let a background task do it.
    # For now, we will sync on read to satisfy the requirement if the DB is empty
    registry.sync_with_db(db)
    sources = registry.get_all_sources(db)

    # Return basic info
    return [
        {
            "id": s.id,
            "slug": s.slug,
            "name": s.name,
            "language": s.language,
            "scrape_type": s.scrape_type,
            "is_active": s.is_active,
            "is_healthy": s.is_healthy,
        }
        for s in sources
    ]


@router.get("/health")
def get_source_health(db: Session = Depends(get_db)):
    """Get health status of all sources."""
    sources = registry.get_all_sources(db)
    return [
        {
            "slug": s.slug,
            "name": s.name,
            "is_healthy": s.is_healthy,
            "last_scrape_time": s.last_scrape_time,
            "success_count": s.success_count,
            "error_count": s.error_count,
        }
        for s in sources
    ]


@router.post("")
def add_source(config: SourceConfig, db: Session = Depends(get_db)):
    """Add or update a source configuration."""
    slug = config.name.lower().replace(" ", "_")

    # Save as JSON file
    file_path = registry.config_dir / f"{slug}.json"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(mode="json"), f, indent=4)

        # Reload and sync
        registry.load_from_disk()
        registry.sync_with_db(db)

        return {"status": "success", "message": f"Config saved to {file_path.name}"}
    except Exception as e:
        logger.error(f"Error saving source config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save source configuration")
