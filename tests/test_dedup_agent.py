"""Tests for Embedding-based Deduplication Agent (Issue 9).

Acceptance Criteria covered:
  AC1 - Embeddings generated using sentence-transformers (all-MiniLM-L6-v2)
  AC2 - Cosine similarity > 0.85 correctly identifies paraphrased duplicates
  AC3 - Unique articles (similarity < 0.85) pass through
  AC4 - Duplicate articles linked to primary via "also_reported_by" field
  AC5 - Primary article selected from highest-priority source
  AC6 - Comparison window is configurable (default 48 hours)
  AC7 - Batch processing handles 100+ articles per cycle efficiently (< 5 seconds)
  AC8 - Embeddings stored in database for reuse
"""

import time
import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from src.agents.dedup_agent import DedupAgent, DedupResult
from src.config.settings import Language
from src.models.article import Article, ArticleEmbedding
from src.models.category import Category
from src.models.source import Source

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_engine():
    """Create an in-memory SQLite engine with StaticPool for thread safety."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Category.__table__.create(engine)
    Source.__table__.create(engine)
    Article.__table__.create(engine)
    ArticleEmbedding.__table__.create(engine)
    return engine


@pytest.fixture
def db_session(shared_engine):
    """Provide a clean DB session."""
    session_factory = sessionmaker(bind=shared_engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def dedup_agent():
    """Shared DedupAgent instance."""
    return DedupAgent()


# ---------------------------------------------------------------------------
# AC1 - Embeddings generated using sentence-transformers (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------


class TestEmbeddingGeneration:
    def test_embeddings_dimensions_and_norm(self, dedup_agent):
        """AC1: Embeddings are generated with shape (N, 384) and L2 normalized."""
        texts = [
            "ISRO successfully launches Chandrayaan-3 mission to the Moon.",
            "Stock market reaches an all-time peak with Sensex gaining 500 points.",
        ]
        embeddings = dedup_agent.generate_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384), "all-MiniLM-L6-v2 should produce 384-dimensional embeddings"

        # Check normalization (norm ~ 1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-3)

    def test_empty_texts_returns_empty_array(self, dedup_agent):
        """AC1: Empty inputs produce empty arrays safely."""
        embeddings = dedup_agent.generate_embeddings([])
        assert embeddings.shape == (0, 384)


# ---------------------------------------------------------------------------
# AC2 & AC3 - Semantic Deduplication & Unique Pass-through
# ---------------------------------------------------------------------------


class TestSemanticDeduplication:
    def test_paraphrased_duplicates_identified(self, dedup_agent):
        """AC2 & AC3: Paraphrased duplicates (> 0.85) caught, unique articles pass through."""
        article1 = {
            "title": "India wins cricket match against Australia by 5 wickets",
            "body": "India defeated Australia by 5 wickets in a thrilling final match at Mumbai stadium.",
            "source_slug": "ndtv",
        }
        article2 = {
            "title": "India beats Australia in thrilling cricket match",
            "body": "In a thrilling encounter at Mumbai stadium, India beat Australia by 5 wickets in the final.",
            "source_slug": "timesofindia",
        }
        article3 = {
            "title": "Stock market hits record high with Sensex above 80000",
            "body": "Indian benchmark indices soared to fresh record highs led by banking and IT shares.",
            "source_slug": "mint",
        }

        result = dedup_agent.check_duplicates([article1, article2, article3])

        assert isinstance(result, DedupResult)
        assert len(result.unique) == 2, "Expected 2 unique stories (Cricket & Stock Market)"
        assert len(result.duplicates) == 1, "Expected 1 duplicate cricket article"

        # Ensure article3 (stocks) is in unique
        assert any("Stock market" in a["title"] for a in result.unique)

    def test_identical_articles_detected(self, dedup_agent):
        """AC2: Identical articles are identified with similarity ~ 1.0."""
        art1 = {"title": "Breaking: Union Budget 2026 presented", "body": "Finance Minister presented the budget."}
        art2 = {"title": "Breaking: Union Budget 2026 presented", "body": "Finance Minister presented the budget."}

        result = dedup_agent.check_duplicates([art1, art2])
        assert len(result.unique) == 1
        assert len(result.duplicates) == 1


# ---------------------------------------------------------------------------
# AC4 - Duplicate articles linked via "also_reported_by"
# ---------------------------------------------------------------------------


class TestDuplicateLinking:
    def test_also_reported_by_linking(self, dedup_agent):
        """AC4: Duplicate source added to also_reported_by list of the primary article."""
        article_hindu = {
            "id": str(uuid.uuid4()),
            "title": "India wins cricket match against Australia by 5 wickets",
            "body": "India defeated Australia by 5 wickets in a thrilling match at Mumbai stadium.",
            "source_slug": "thehindu",
        }
        article_ht = {
            "id": str(uuid.uuid4()),
            "title": "India beats Australia in thrilling cricket match",
            "body": "India beat Australia by 5 wickets in a thrilling match at Mumbai stadium.",
            "source_slug": "hindustantimes",
        }

        result = dedup_agent.check_duplicates([article_hindu, article_ht])

        assert len(result.unique) == 1
        assert len(result.duplicates) == 1

        primary = result.unique[0]
        duplicate = result.duplicates[0]

        # Check also_reported_by
        assert "hindustantimes" in primary["also_reported_by"]
        assert duplicate["is_duplicate"] is True
        assert duplicate["primary_article_id"] == primary["id"]


# ---------------------------------------------------------------------------
# AC5 - Primary article selected from highest-priority source
# ---------------------------------------------------------------------------


class TestSourcePrioritySelection:
    def test_highest_priority_source_becomes_primary(self, dedup_agent):
        """AC5: The Hindu (prio 100) is preferred over Times of India (prio 80)."""
        # Place lower priority first in list to verify order independence
        article_toi = {
            "id": "toi-123",
            "title": "India wins cricket match against Australia by 5 wickets",
            "body": "India defeated Australia by 5 wickets in a thrilling match at Mumbai.",
            "source_slug": "timesofindia",
        }
        article_hindu = {
            "id": "hindu-456",
            "title": "India beats Australia in thrilling cricket match by 5 wickets",
            "body": "India beat Australia by 5 wickets in a thrilling cricket match at Mumbai.",
            "source_slug": "thehindu",
        }

        result = dedup_agent.check_duplicates([article_toi, article_hindu])

        assert len(result.unique) == 1
        primary = result.unique[0]
        assert primary["source_slug"] == "thehindu", "The Hindu should be chosen as primary over Times of India"
        assert primary["id"] == "hindu-456"

        duplicate = result.duplicates[0]
        assert duplicate["source_slug"] == "timesofindia"
        assert duplicate["primary_article_id"] == "hindu-456"


# ---------------------------------------------------------------------------
# AC6 - Configurable comparison window
# ---------------------------------------------------------------------------


class TestConfigurableComparisonWindow:
    def test_window_hours_filtering(self, dedup_agent, db_session):
        """AC6: Only articles within the window_hours are compared against incoming articles."""
        cat = Category(id=10, slug="cricket", name="Cricket", is_active=True)
        db_session.merge(cat)
        src = Source(
            id=uuid.uuid4(),
            slug="thehindu",
            name="The Hindu",
            base_url="https://thehindu.com",
            article_list_url="https://thehindu.com",
            language=Language.ENGLISH,
        )
        db_session.merge(src)
        db_session.commit()

        # 1. Create an old article (72 hours ago) with embedding
        old_id = uuid.uuid4()
        old_article = Article(
            id=old_id,
            title="India wins cricket match against Australia by 5 wickets",
            body="India defeated Australia by 5 wickets in thrilling match at Mumbai stadium.",
            source_id=src.id,
            category_id=cat.id,
            language=Language.ENGLISH,
            source_url=f"https://thehindu.com/old-cricket-{old_id.hex}",
            publish_time=datetime.utcnow() - timedelta(hours=72),
            is_duplicate=False,
        )
        db_session.add(old_article)
        db_session.flush()

        old_emb = dedup_agent.generate_embeddings([f"{old_article.title}. {old_article.body}"])[0].tolist()
        db_session.add(ArticleEmbedding(article_id=old_id, embedding=old_emb))
        db_session.commit()

        # 2. Incoming article about the exact same match
        incoming = {
            "title": "India beats Australia in thrilling cricket match by 5 wickets",
            "body": "India beat Australia by 5 wickets in thrilling match at Mumbai stadium.",
            "source_slug": "ndtv",
        }

        # With standard 48 hour window, the 72 hour old article is ignored -> incoming is unique
        result_48h = dedup_agent.check_duplicates([incoming], db=db_session, window_hours=48)
        assert len(result_48h.unique) == 1
        assert len(result_48h.duplicates) == 0

        # With 96 hour window, old article is considered -> incoming is flagged as duplicate
        result_96h = dedup_agent.check_duplicates([incoming], db=db_session, window_hours=96)
        assert len(result_96h.duplicates) == 1


# ---------------------------------------------------------------------------
# AC7 - Batch processing handles 100+ articles in < 5 seconds
# ---------------------------------------------------------------------------


class TestBatchProcessingPerformance:
    def test_100_articles_processed_under_5_seconds(self, dedup_agent):
        """AC7: Batch processing handles 100+ articles in < 5 seconds."""
        articles = []
        base_topics = [
            ("India wins cricket cup", "The national cricket team defeated England in the semi-final."),
            ("AI advances in healthcare", "New AI diagnostic systems detect diseases earlier."),
            ("Monsoon rains arrive in Kerala", "Southwest monsoon has officially made landfall."),
            ("RBI changes interest rate", "Central bank announced monetary policy committee decision."),
            ("New electric car launched", "Automaker introduced their new long-range electric vehicle."),
        ]

        # Generate 100 variations (20 of each topic)
        for i in range(100):
            topic_title, topic_body = base_topics[i % len(base_topics)]
            articles.append(
                {
                    "id": f"art-{i}",
                    "title": f"{topic_title} - update #{i}",
                    "body": f"{topic_body} Extra information for article {i}.",
                    "source_slug": "ndtv" if i % 2 == 0 else "timesofindia",
                }
            )

        start_time = time.time()
        result = dedup_agent.check_duplicates(articles)
        elapsed = time.time() - start_time

        assert len(articles) == 100
        assert len(result.all_articles) == 100
        assert elapsed < 5.0, f"Processing 100 articles took {elapsed:.2f}s, must be < 5.0s"
        # Should detect clusters for the 5 topics
        assert len(result.unique) <= 10


# ---------------------------------------------------------------------------
# AC8 - Embeddings stored in database for reuse
# ---------------------------------------------------------------------------


class TestEmbeddingDatabaseStorage:
    def test_embeddings_stored_in_db_table(self, dedup_agent, db_session):
        """AC8: Article embeddings are persisted to article_embeddings table."""
        cat = Category(id=20, slug="technology", name="Technology", is_active=True)
        db_session.merge(cat)
        src = Source(
            id=uuid.uuid4(),
            slug="ndtv",
            name="NDTV",
            base_url="https://ndtv.com",
            article_list_url="https://ndtv.com",
            language=Language.ENGLISH,
        )
        db_session.merge(src)
        db_session.commit()

        art_id = uuid.uuid4()
        art = Article(
            id=art_id,
            title="OpenAI announces new AI reasoning model",
            body="A next-generation reasoning model has been announced with strong coding skills.",
            source_id=src.id,
            category_id=cat.id,
            language=Language.ENGLISH,
            source_url=f"https://ndtv.com/tech-{art_id.hex}",
            publish_time=datetime.utcnow(),
            is_duplicate=False,
        )
        db_session.add(art)
        db_session.commit()

        # Run dedup with db session
        result = dedup_agent.check_duplicates([art], db=db_session)
        assert len(result.unique) == 1

        # Check article_embeddings table
        stored_emb = db_session.query(ArticleEmbedding).filter(ArticleEmbedding.article_id == art_id).first()
        assert stored_emb is not None
        assert len(stored_emb.embedding) == 384
        assert isinstance(stored_emb.embedding[0], float)
