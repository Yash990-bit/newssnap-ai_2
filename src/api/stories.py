"""NewsSnap AI - Story API Router (Issue 10).

Endpoints:
- GET /api/stories/{id}: Get story by ID with all related articles
- GET /api/stories: List recent stories
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.models.article import Article
from src.models.story import Story

router = APIRouter(prefix="/api/stories", tags=["stories"])


def _format_article(article: Optional[Article]) -> Optional[Dict[str, Any]]:
    """Helper to format an Article model into a dictionary."""
    if not article:
        return None
    src = article.source
    cat = article.category
    return {
        "id": str(article.id),
        "title": article.title,
        "body": article.body,
        "source_url": article.source_url,
        "image_url": article.image_url,
        "publish_time": article.publish_time.isoformat() if article.publish_time else None,
        "author": article.author,
        "quality_score": article.quality_score,
        "language": article.language.value if hasattr(article.language, "value") else str(article.language),
        "source_slug": src.slug if src else None,
        "source_name": src.name if src else None,
        "category": cat.slug if cat else "national",
    }


@router.get("/{story_id}")
def get_story(story_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get a specific story by UUID, including its primary article and all related articles.
    """
    try:
        story_uuid = uuid.UUID(story_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid story ID format")

    story = db.query(Story).filter(Story.id == story_uuid).first()
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")

    # Fetch primary article
    primary = story.primary_article

    # Fetch all related articles (articles with story_id == story.id but not primary_article_id)
    all_story_articles = (
        db.query(Article)
        .filter(
            Article.story_id == story.id,
            Article.id != story.primary_article_id,
        )
        .all()
    )

    formatted_primary = _format_article(primary)
    formatted_related = [_format_article(a) for a in all_story_articles]

    return {
        "id": str(story.id),
        "primary_article": formatted_primary,
        "related_articles": formatted_related,
        "article_count": 1 + len(formatted_related) if formatted_primary else len(formatted_related),
        "created_at": story.created_at.isoformat() if story.created_at else None,
    }


@router.get("")
def list_stories(
    limit: int = Query(default=20, ge=1, le=100, description="Max stories to return"),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    List recently clustered stories.
    """
    stories = db.query(Story).order_by(Story.created_at.desc()).limit(limit).all()

    results = []
    for s in stories:
        primary = s.primary_article
        related_count = (
            db.query(Article)
            .filter(
                Article.story_id == s.id,
                Article.id != s.primary_article_id,
            )
            .count()
        )
        results.append(
            {
                "id": str(s.id),
                "primary_article": _format_article(primary),
                "related_count": related_count,
                "total_articles": 1 + related_count if primary else related_count,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
        )
    return results
