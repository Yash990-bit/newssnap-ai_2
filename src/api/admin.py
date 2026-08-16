"""NewsSnap AI - Admin API (Issue 12).

Provides admin-only endpoints for pipeline monitoring:
  GET /api/admin/rejected-articles  — recent articles rejected by QualityFilter
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from src.agents.quality_filter import QualityFilter, RejectionRecord

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
