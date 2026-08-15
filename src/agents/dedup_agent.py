"""NewsSnap AI - Embedding-based Deduplication Agent (Issue 9).

Identifies and links paraphrased duplicate news articles across multiple sources:
- Generates dense semantic embeddings using sentence-transformers (all-MiniLM-L6-v2)
- Compares embeddings using cosine similarity (> 0.85 indicates a duplicate)
- Retains the primary article from the highest-priority source
- Links duplicates to the primary article via 'also_reported_by' and 'primary_article_id'
- Supports configurable comparison windows (default 48 hours) and similarity thresholds
- High performance batch processing (< 5s for 100+ articles)
- Stores and reuses embeddings in the database
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from src.models.article import Article, ArticleEmbedding
from src.models.source import Source

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_WINDOW_HOURS = 48

# Default source priority rankings (higher score = preferred as primary article)
DEFAULT_SOURCE_PRIORITY: Dict[str, int] = {
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
class DedupResult:
    """Result returned by the deduplication agent."""

    unique: List[Any] = field(default_factory=list)
    duplicates: List[Any] = field(default_factory=list)
    all_articles: List[Any] = field(default_factory=list)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)

    @property
    def unique_count(self) -> int:
        return len(self.unique)


class DedupAgent:
    """Agent for detecting and linking semantically duplicate articles using neural embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        source_priorities: Optional[Dict[str, int]] = None,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.window_hours = window_hours
        self.source_priorities = source_priorities or DEFAULT_SOURCE_PRIORITY
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-loaded SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ----------------------------------------------------------------------
    # Embedding Generation & Similarity
    # ----------------------------------------------------------------------

    def generate_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate normalized embeddings for a list of text strings.

        Args:
            texts: List of strings (e.g. "title. body").
            batch_size: Batch size for encoding.

        Returns:
            2D numpy array of shape (N, embedding_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings

    def compute_similarity(self, embeddings_a: np.ndarray, embeddings_b: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cosine similarity matrix between two sets of embeddings.

        Args:
            embeddings_a: Shape (N, D).
            embeddings_b: Shape (M, D). If None, computes pairwise similarity for embeddings_a.

        Returns:
            Cosine similarity matrix of shape (N, M) or (N, N).
        """
        if embeddings_a.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        if embeddings_b is None:
            return cosine_similarity(embeddings_a, embeddings_a)
        else:
            if embeddings_b.size == 0:
                return np.empty((embeddings_a.shape[0], 0), dtype=np.float32)
            return cosine_similarity(embeddings_a, embeddings_b)

    # ----------------------------------------------------------------------
    # Helper extractors for polymorphic article structures (Dict, Model, Dataclass)
    # ----------------------------------------------------------------------

    def _get_article_text(self, item: Any) -> str:
        """Extract title and leading body content for embedding."""
        if isinstance(item, dict):
            title = item.get("title", "")
            body = item.get("body", "")
        else:
            title = getattr(item, "title", "")
            body = getattr(item, "body", "")

        # Limit body length for faster tokenization and high semantic focus
        body_snippet = body[:1000] if body else ""
        return f"{title}. {body_snippet}".strip()

    def _get_article_source(self, item: Any) -> str:
        """Extract source slug or source identifier."""
        if isinstance(item, dict):
            return item.get("source_slug") or item.get("source") or ""
        source = getattr(item, "source", None)
        if isinstance(source, Source):
            return source.slug
        if isinstance(source, str):
            return source
        return getattr(item, "source_slug", "") or ""

    def _get_source_priority(self, item: Any) -> int:
        """Get priority score for the source of an article."""
        slug = self._get_article_source(item).lower().replace("-", "_").replace(" ", "_")
        return self.source_priorities.get(slug, 50)

    def _get_article_id(self, item: Any) -> Optional[Any]:
        """Extract or generate ID of an article."""
        if isinstance(item, dict):
            return item.get("id")
        return getattr(item, "id", None)

    # ----------------------------------------------------------------------
    # Core Deduplication Logic
    # ----------------------------------------------------------------------

    def check_duplicates(
        self,
        articles: List[Any],
        db: Optional[Session] = None,
        threshold: Optional[float] = None,
        window_hours: Optional[int] = None,
    ) -> DedupResult:
        """
        Deduplicate a list of incoming articles against each other and against historical DB articles.

        Args:
            articles: List of articles (dicts, ORM Article instances, or ScrapedArticle objects).
            db: Optional SQLAlchemy Session for fetching historical articles and storing embeddings.
            threshold: Optional cosine similarity threshold override (default 0.85).
            window_hours: Optional lookback window override (default 48 hours).

        Returns:
            DedupResult containing unique and duplicate articles with relationships linked.
        """
        if not articles:
            return DedupResult()

        sim_threshold = threshold if threshold is not None else self.similarity_threshold
        win_hours = window_hours if window_hours is not None else self.window_hours

        # 1. Generate embeddings for all incoming articles in batch
        texts = [self._get_article_text(art) for art in articles]
        incoming_embeddings = self.generate_embeddings(texts)

        # 2. Optionally fetch historical articles & embeddings from DB within window_hours
        historical_articles: List[Article] = []
        historical_embeddings_list: List[np.ndarray] = []

        if db is not None:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=win_hours)
                cutoff_naive = cutoff_time.replace(tzinfo=None)

                # Query non-duplicate primary articles within the time window
                db_articles = (
                    db.query(Article)
                    .filter(
                        Article.publish_time >= cutoff_naive,
                        Article.is_duplicate.is_(False),
                    )
                    .all()
                )

                for db_art in db_articles:
                    if db_art.embedding and db_art.embedding.embedding:
                        historical_articles.append(db_art)
                        historical_embeddings_list.append(np.array(db_art.embedding.embedding, dtype=np.float32))
            except Exception as e:
                logger.warning(f"Failed to fetch historical articles for dedup: {e}")

        # Structure to track status of each incoming article
        # Each entry: { "article": art, "is_duplicate": False, "primary": None, "also_reported_by": [] }
        records = []
        for art in articles:
            rec = {
                "article": art,
                "is_duplicate": False,
                "primary": None,
                "also_reported_by": [],
                "similarity": 0.0,
            }
            # Initialize also_reported_by on dict / object if present
            if isinstance(art, dict):
                art.setdefault("also_reported_by", [])
                art["is_duplicate"] = False
            else:
                if not hasattr(art, "also_reported_by"):
                    setattr(art, "also_reported_by", [])
                setattr(art, "is_duplicate", False)
            records.append(rec)

        # 3. Check against historical articles if available
        if historical_articles and historical_embeddings_list:
            hist_emb_matrix = np.array(historical_embeddings_list, dtype=np.float32)
            hist_sim_matrix = self.compute_similarity(incoming_embeddings, hist_emb_matrix)

            for i, rec in enumerate(records):
                sims = hist_sim_matrix[i]
                best_hist_idx = int(np.argmax(sims))
                best_sim = float(sims[best_hist_idx])

                if best_sim >= sim_threshold:
                    hist_primary = historical_articles[best_hist_idx]
                    rec["is_duplicate"] = True
                    rec["primary"] = hist_primary
                    rec["similarity"] = best_sim
                    src_name = self._get_article_source(rec["article"]) or "another source"

                    self._mark_duplicate(
                        duplicate_item=rec["article"],
                        primary_item=hist_primary,
                        source_name=src_name,
                    )

        # 4. Intra-batch deduplication among incoming articles
        batch_sim_matrix = self.compute_similarity(incoming_embeddings)
        n = len(records)

        for i in range(n):
            if records[i]["is_duplicate"]:
                continue

            for j in range(i + 1, n):
                if records[j]["is_duplicate"]:
                    continue

                sim = float(batch_sim_matrix[i, j])
                if sim >= sim_threshold:
                    # Decide primary between article i and article j based on source priority
                    prio_i = self._get_source_priority(records[i]["article"])
                    prio_j = self._get_source_priority(records[j]["article"])

                    if prio_i >= prio_j:
                        primary_rec = records[i]
                        dup_rec = records[j]
                    else:
                        primary_rec = records[j]
                        dup_rec = records[i]

                    dup_rec["is_duplicate"] = True
                    dup_rec["primary"] = primary_rec["article"]
                    dup_rec["similarity"] = sim

                    dup_src_name = self._get_article_source(dup_rec["article"]) or "another source"
                    self._mark_duplicate(
                        duplicate_item=dup_rec["article"],
                        primary_item=primary_rec["article"],
                        source_name=dup_src_name,
                    )

                    # If i became duplicate, stop comparing i against other items
                    if dup_rec is records[i]:
                        break

        # 5. Optionally save embeddings for new articles to database
        if db is not None:
            try:
                for i, rec in enumerate(records):
                    art = rec["article"]
                    if isinstance(art, Article) and art.id:
                        self._save_single_embedding(db, art.id, incoming_embeddings[i].tolist())
                db.commit()
            except Exception as e:
                logger.error(f"Error persisting embeddings to database: {e}")
                db.rollback()

        unique_list = [r["article"] for r in records if not r["is_duplicate"]]
        duplicates_list = [r["article"] for r in records if r["is_duplicate"]]

        logger.info(
            f"Dedup completed: {len(unique_list)} unique, {len(duplicates_list)} duplicates "
            f"out of {len(articles)} articles (threshold={sim_threshold})."
        )

        return DedupResult(
            unique=unique_list,
            duplicates=duplicates_list,
            all_articles=[r["article"] for r in records],
        )

    # ----------------------------------------------------------------------
    # Helper Linking Methods
    # ----------------------------------------------------------------------

    def _mark_duplicate(self, duplicate_item: Any, primary_item: Any, source_name: str) -> None:
        """Link duplicate item to primary item via also_reported_by and IDs."""
        primary_id = self._get_article_id(primary_item)

        # Update duplicate
        if isinstance(duplicate_item, dict):
            duplicate_item["is_duplicate"] = True
            if primary_id is not None:
                duplicate_item["primary_article_id"] = primary_id
            duplicate_item["primary_article"] = primary_item
        else:
            setattr(duplicate_item, "is_duplicate", True)
            if primary_id is not None:
                setattr(duplicate_item, "primary_article_id", primary_id)
            setattr(duplicate_item, "primary_article", primary_item)

        # Update primary with also_reported_by
        if isinstance(primary_item, dict):
            also_reported = primary_item.setdefault("also_reported_by", [])
            if source_name and source_name not in also_reported:
                also_reported.append(source_name)
        else:
            also_reported = getattr(primary_item, "also_reported_by", None)
            if also_reported is None:
                also_reported = []
                setattr(primary_item, "also_reported_by", also_reported)
            if source_name and source_name not in also_reported:
                also_reported.append(source_name)

    def _save_single_embedding(self, db: Session, article_id: uuid.UUID, embedding_vec: List[float]) -> None:
        """Store or update embedding vector for an article in DB."""
        existing = db.query(ArticleEmbedding).filter(ArticleEmbedding.article_id == article_id).first()
        if existing:
            existing.embedding = embedding_vec
        else:
            new_emb = ArticleEmbedding(
                article_id=article_id,
                embedding=embedding_vec,
            )
            db.add(new_emb)
