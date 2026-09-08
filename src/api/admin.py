"""NewsSnap AI - Admin and Health Monitoring API (Issue 8 & Issue 12).

Provides admin-only endpoints for pipeline monitoring:
  GET /api/admin/scrape-health: per-source health metrics and pipeline throughput
  GET /api/admin/recent-articles: recent articles fetched by the scraper
  POST /api/admin/trigger-scrape: manually trigger a scrape cycle
  GET /api/admin/rejected-articles: recent articles rejected by QualityFilter
  DELETE /api/admin/rejected-articles/clear: clear in-memory rejection log
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.agents.quality_filter import QualityFilter, RejectionRecord
from src.config.database import get_db
from src.scheduler.scrape_pipeline import pipeline_instance

router = APIRouter(prefix="/api/admin", tags=["admin"])

# ---------------------------------------------------------------------------
# In-process singleton filter instance (shared with the pipeline).
# In a real deployment this would be backed by a DB table; for now we use
# the filter's in-memory log so the endpoint works without a running DB.
# ---------------------------------------------------------------------------
_filter_instance: Optional[QualityFilter] = None


def get_filter() -> QualityFilter:
    """Return (or lazily create) the shared QualityFilter instance."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = QualityFilter()
    return _filter_instance


def _record_to_dict(record: RejectionRecord) -> Dict[str, Any]:
    return {
        "article_id": record.article_id,
        "source": record.source,
        "title": record.title,
        "rejection_reason": record.rejection_reason,
        "score": record.score,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/scrape-health")
def get_scrape_health(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get comprehensive health metrics for the news scraping pipeline.

    Returns per-source health status, success/error counts, cooldown status, and throughput.
    """
    return pipeline_instance.get_health_metrics(db=db)


@router.get("/recent-articles")
def get_recent_articles(
    limit: int = Query(default=10, ge=1, le=100, description="Max number of articles to return"),
    source_slug: Optional[str] = Query(default=None, description="Filter by source slug"),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get the most recently scraped articles stored in the database."""
    return pipeline_instance.get_recent_articles(db=db, limit=limit, source_slug=source_slug)


@router.post("/trigger-scrape")
async def trigger_scrape(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Manually trigger an immediate scraping cycle across all sources."""
    result = await pipeline_instance.run_scrape_cycle(db=db)
    return {"status": "completed", "result": result}


@router.get("/rejected-articles")
def get_rejected_articles(
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
) -> Dict[str, Any]:
    """Return recent articles rejected by the quality filter.

    Returns:
        JSON with total count and list of rejection records (most-recent first).
    """
    qf = get_filter()
    log = qf.get_rejection_log()
    # Most recent first, truncated to limit
    recent = log[-limit:][::-1]
    return {
        "total": len(log),
        "returned": len(recent),
        "rejections": [_record_to_dict(r) for r in recent],
    }


@router.delete("/rejected-articles/clear")
def clear_rejected_articles() -> Dict[str, str]:
    """Clear the in-memory rejection log (admin use only)."""
    get_filter().clear_log()
    return {"status": "cleared"}
