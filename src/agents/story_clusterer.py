"""NewsSnap AI - Story Clustering Agent (Issue 10).

Clusters related news articles covering the same story:
- Uses DBSCAN on dense semantic embeddings (all-MiniLM-L6-v2, eps=0.3, min_samples=2)
- Each story has exactly one primary article and 0+ related articles
- Selects primary article based on composite quality score (source priority, length, lead image)
- Merges new incoming articles into existing active stories
- Standalone/singleton articles become individual single-article stories
- High performance clustering: < 10 seconds for 500+ articles
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sqlalchemy.orm import Session

from src.models.article import Article
from src.models.source import Source
from src.models.story import Story, StoryArticle

logger = logging.getLogger(__name__)

# Default clustering parameters
DEFAULT_EPS = 0.3
DEFAULT_MIN_SAMPLES = 2
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Source quality/priority ranking (0 - 100)
SOURCE_PRIORITIES: Dict[str, int] = {
    "thehindu": 100,
    "the_hindu": 100,
    "indianexpress": 95,
    "indian_express": 95,
    "ndtv": 90,
    "hindustantimes": 85,
    "hindustan_times": 85,
    "mint": 85,
    "timesofindia": 80,
    "times_of_india": 80,
    "aajtak": 75,
    "aaj_tak": 75,
    "dainikbhaskar": 70,
    "dainik_bhaskar": 70,
    "news18": 70,
    "zeenews_hindi": 65,
    "zeenews": 65,
}


@dataclass
class StoryCluster:
    """Represents a clustered story containing a primary article and related articles."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    primary_article: Any = None
    related_articles: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def article_count(self) -> int:
        return 1 + len(self.related_articles) if self.primary_article else len(self.related_articles)

    @property
    def all_articles(self) -> List[Any]:
        if self.primary_article:
            return [self.primary_article] + self.related_articles
        return list(self.related_articles)


class StoryClusterer:
    """Agent for clustering semantically related news articles into unified stories using DBSCAN."""

    def __init__(
        self,
        eps: float = DEFAULT_EPS,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        model_name: str = DEFAULT_MODEL_NAME,
        source_priorities: Optional[Dict[str, int]] = None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.model_name = model_name
        self.source_priorities = source_priorities or SOURCE_PRIORITIES
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model for story clustering: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ----------------------------------------------------------------------
    # Quality Scoring for Primary Article Selection
    # ----------------------------------------------------------------------

    def compute_quality_score(self, item: Any) -> float:
        """
        Calculate composite quality score for an article (0.0 to 100.0).
        Factors:
          1. Source Priority: max 40 pts
          2. Content Length (word count): max 40 pts (reaches max at 500 words)
          3. Image Presence: 20 pts (if image is valid/present)
        """
        # 1. Source priority
        source_slug = self._get_source_slug(item).lower().replace("-", "_").replace(" ", "_")
        priority = self.source_priorities.get(source_slug, 50)
        source_score = (priority / 100.0) * 40.0

        # 2. Content length
        body = self._get_body(item)
        word_count = len(body.split()) if body else 0
        length_score = min(word_count / 500.0, 1.0) * 40.0

        # 3. Image presence
        image_url = self._get_image_url(item)
        image_score = 20.0 if image_url and len(image_url.strip()) > 5 else 0.0

        total_score = round(source_score + length_score + image_score, 2)

        # Store quality score on article if attribute or key exists
        if isinstance(item, dict):
            item["quality_score"] = total_score
        elif hasattr(item, "quality_score"):
            setattr(item, "quality_score", total_score)

        return total_score

    # ----------------------------------------------------------------------
    # Helper extractors for polymorphic article structures
    # ----------------------------------------------------------------------

    def _get_title(self, item: Any) -> str:
        if isinstance(item, dict):
            return item.get("title", "")
        return getattr(item, "title", "") or ""

    def _get_body(self, item: Any) -> str:
        if isinstance(item, dict):
            return item.get("body", "")
        return getattr(item, "body", "") or ""

    def _get_image_url(self, item: Any) -> Optional[str]:
        if isinstance(item, dict):
            return item.get("image_url")
        return getattr(item, "image_url", None)

    def _get_source_slug(self, item: Any) -> str:
        if isinstance(item, dict):
            return item.get("source_slug") or item.get("source") or ""
        source = getattr(item, "source", None)
        if isinstance(source, Source):
            return source.slug
        if isinstance(source, str):
            return source
        return getattr(item, "source_slug", "") or ""

    def _get_text_for_embedding(self, item: Any) -> str:
        title = self._get_title(item)
        body = self._get_body(item)[:1000]
        return f"{title}. {body}".strip()

    # ----------------------------------------------------------------------
    # Core Clustering Method
    # ----------------------------------------------------------------------

    def cluster_articles(
        self,
        articles: List[Any],
        existing_stories: Optional[List[StoryCluster]] = None,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> List[StoryCluster]:
        """
        Group articles into Story clusters using DBSCAN on article embeddings.
        - Merges into existing stories if match found within eps threshold.
        - Unclustered/singleton articles become standalone individual stories.
        - Selects highest quality score article as the primary article in each story.
        """
        if not articles:
            return existing_stories or []

        cluster_eps = eps if eps is not None else self.eps
        cluster_min_samples = min_samples if min_samples is not None else self.min_samples

        # 1. Compute embeddings for incoming articles
        texts = [self._get_text_for_embedding(art) for art in articles]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        stories: List[StoryCluster] = list(existing_stories or [])
        unmerged_indices: List[int] = []

        # 2. Check each new article against existing stories (Story Merging)
        if stories:
            # Build story primary embeddings
            story_primary_texts = [self._get_text_for_embedding(s.primary_article) for s in stories]
            story_embeddings = self.model.encode(
                story_primary_texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            # Compute pairwise distance between new articles and existing stories
            dist_matrix = cosine_distances(embeddings, story_embeddings)

            for i, art in enumerate(articles):
                best_story_idx = int(np.argmin(dist_matrix[i]))
                best_dist = float(dist_matrix[i, best_story_idx])

                if best_dist <= cluster_eps:
                    target_story = stories[best_story_idx]
                    art_score = self.compute_quality_score(art)
                    primary_score = self.compute_quality_score(target_story.primary_article)

                    if art_score > primary_score:
                        # New article is higher quality: make it primary and demote old primary
                        old_primary = target_story.primary_article
                        target_story.primary_article = art
                        target_story.related_articles.append(old_primary)
                    else:
                        target_story.related_articles.append(art)
                else:
                    unmerged_indices.append(i)
        else:
            unmerged_indices = list(range(len(articles)))

        # 3. Cluster remaining unmerged articles using DBSCAN
        if unmerged_indices:
            subset_embeddings = embeddings[unmerged_indices]
            subset_articles = [articles[i] for i in unmerged_indices]

            if len(subset_articles) == 1:
                # Exactly 1 article: standalone story
                art = subset_articles[0]
                self.compute_quality_score(art)
                stories.append(StoryCluster(primary_article=art, related_articles=[]))
            else:
                dbscan = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples, metric="cosine")
                labels = dbscan.fit_predict(subset_embeddings)

                # Group articles by cluster label
                clusters_map: Dict[int, List[Any]] = {}
                for idx, label in enumerate(labels):
                    clusters_map.setdefault(label, []).append(subset_articles[idx])

                # Process formed clusters (label >= 0)
                for label, cluster_items in clusters_map.items():
                    if label == -1:
                        # Noise points: each singleton article becomes a standalone story
                        for singleton_art in cluster_items:
                            self.compute_quality_score(singleton_art)
                            stories.append(StoryCluster(primary_article=singleton_art, related_articles=[]))
                    else:
                        # Multi-article story cluster: select primary article with highest quality score
                        for item in cluster_items:
                            self.compute_quality_score(item)

                        sorted_items = sorted(
                            cluster_items,
                            key=lambda x: self.compute_quality_score(x),
                            reverse=True,
                        )
                        primary = sorted_items[0]
                        related = sorted_items[1:]
                        stories.append(StoryCluster(primary_article=primary, related_articles=related))

        # 4. If database session provided, persist / update Story models
        if db is not None:
            self._persist_stories_to_db(db, stories)

        logger.info(f"Clustered {len(articles)} articles into {len(stories)} stories.")
        return stories

    # ----------------------------------------------------------------------
    # Database Persistence & Recent Articles Clustering
    # ----------------------------------------------------------------------

    def _persist_stories_to_db(self, db: Session, stories: List[StoryCluster]) -> None:
        """Save story clusters and link articles in database."""
        try:
            for sc in stories:
                # Check if primary article is an Article ORM object with an ID
                p_art = sc.primary_article
                p_id = getattr(p_art, "id", None)
                if not p_id or not isinstance(p_art, Article):
                    continue

                story_uuid = uuid.UUID(sc.id) if isinstance(sc.id, str) else sc.id
                db_story = db.query(Story).filter(Story.id == story_uuid).first()

                if not db_story:
                    db_story = Story(
                        id=story_uuid,
                        primary_article_id=p_id,
                    )
                    db.add(db_story)
                    db.flush()

                p_art.story_id = db_story.id

                for r_art in sc.related_articles:
                    if isinstance(r_art, Article) and r_art.id:
                        r_art.story_id = db_story.id
                        # Junction table entry if needed
                        sa_entry = (
                            db.query(StoryArticle)
                            .filter(
                                StoryArticle.story_id == db_story.id,
                                StoryArticle.article_id == r_art.id,
                            )
                            .first()
                        )
                        if not sa_entry:
                            db.add(StoryArticle(story_id=db_story.id, article_id=r_art.id))

            db.commit()
        except Exception as e:
            logger.error(f"Failed to persist stories to DB: {e}")
            db.rollback()

    def cluster_recent_articles(
        self,
        hours: int = 6,
        db: Optional[Session] = None,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> List[StoryCluster]:
        """
        Fetch articles from database within last `hours` and cluster them into stories.
        """
        if db is None:
            logger.warning("No DB session provided to cluster_recent_articles. Returning empty.")
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_naive = cutoff_time.replace(tzinfo=None)

        articles = (
            db.query(Article)
            .filter(
                Article.publish_time >= cutoff_naive,
                Article.is_duplicate.is_(False),
            )
            .all()
        )

        if not articles:
            logger.info(f"No recent articles found in the last {hours} hours.")
            return []

        return self.cluster_articles(articles=articles, db=db, eps=eps, min_samples=min_samples)
