"""Tests for Story Clustering Agent (Issue 10).

Acceptance Criteria covered:
  AC1 - DBSCAN clustering groups related articles into stories
  AC2 - Each story has exactly one primary article and 0+ related articles
  AC3 - Primary article is selected by quality score (source priority, content length, has image)
  AC4 - New articles are checked against existing stories before creating new ones
  AC5 - Singleton articles (no cluster) become standalone stories
  AC6 - GET /api/stories/{id} returns story with all related articles
  AC7 - Clustering runs in < 10 seconds for 500 articles
"""

import time
import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from src.agents.story_clusterer import StoryCluster, StoryClusterer
from src.api.main import app
from src.config.database import get_db
from src.config.settings import Language
from src.models.article import Article, ArticleEmbedding
from src.models.category import Category
from src.models.source import Source
from src.models.story import Story, StoryArticle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Category.__table__.create(eng)
    Source.__table__.create(eng)
    Story.__table__.create(eng)
    Article.__table__.create(eng)
    ArticleEmbedding.__table__.create(eng)
    StoryArticle.__table__.create(eng)
    return eng


@pytest.fixture
def db_session(engine):
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def clusterer():
    return StoryClusterer()


@pytest.fixture(scope="module")
def api_client(engine):
    """TestClient with DB override."""
    factory = sessionmaker(bind=engine)

    def override_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_db, None)


def make_article_dict(title, body, source_slug="ndtv", image_url=None):
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "body": body,
        "source_slug": source_slug,
        "image_url": image_url,
        "quality_score": None,
    }


def make_article_orm(db, src, cat, title, body, image_url=None, quality_score=None):
    art = Article(
        id=uuid.uuid4(),
        title=title,
        body=body,
        source_id=src.id,
        category_id=cat.id,
        language=Language.ENGLISH,
        source_url=f"https://test.com/{uuid.uuid4().hex}",
        publish_time=datetime.utcnow(),
        is_duplicate=False,
        image_url=image_url,
        quality_score=quality_score,
    )
    db.add(art)
    db.flush()
    return art


# ---------------------------------------------------------------------------
# AC1 - DBSCAN groups related articles into stories
# ---------------------------------------------------------------------------


class TestDBSCANClustering:
    def test_related_articles_grouped_into_story(self, clusterer):
        """AC1: DBSCAN clusters semantically related articles into the same story."""
        cricket_1 = make_article_dict(
            title="India wins cricket match against Australia by 5 wickets",
            body="India defeated Australia by 5 wickets in a thrilling final match at Mumbai stadium.",
            source_slug="ndtv",
        )
        cricket_2 = make_article_dict(
            title="India beats Australia in thrilling cricket encounter",
            body="India beat Australia by 5 wickets in a thrilling cricket match at Mumbai.",
            source_slug="timesofindia",
        )
        markets_1 = make_article_dict(
            title="Stock market hits record high with Sensex above 80000",
            body="Indian benchmark indices soared to fresh record highs led by banking stocks.",
            source_slug="mint",
        )

        stories = clusterer.cluster_articles([cricket_1, cricket_2, markets_1])

        assert len(stories) == 2, "Expected 2 stories: cricket and markets"
        all_counts = sorted([s.article_count for s in stories], reverse=True)
        assert all_counts[0] == 2, "Cricket story should have 2 articles"
        assert all_counts[1] == 1, "Markets story should have 1 article"

    def test_different_topics_not_grouped(self, clusterer):
        """AC1: Distinct topics produce separate stories."""
        articles = [
            make_article_dict("ISRO launches satellite successfully", "India's space program scored a win."),
            make_article_dict("Budget 2026 presented by Finance Minister", "FM announced income tax cuts."),
            make_article_dict("Monsoon expected early this year", "Met department forecasts timely rains."),
        ]
        stories = clusterer.cluster_articles(articles)
        assert len(stories) == 3, "Each distinct topic should produce its own story"


# ---------------------------------------------------------------------------
# AC2 - Each story has exactly one primary and 0+ related articles
# ---------------------------------------------------------------------------


class TestStoryStructure:
    def test_story_has_primary_and_related(self, clusterer):
        """AC2: Each story has exactly one primary article and 0+ related."""
        article1 = make_article_dict(
            "India wins cricket match against Australia by 5 wickets",
            "India defeated Australia by 5 wickets at Mumbai.",
            source_slug="thehindu",
        )
        article2 = make_article_dict(
            "India beats Australia in cricket final",
            "India beat Australia by five wickets in the final at Mumbai.",
            source_slug="ndtv",
        )

        stories = clusterer.cluster_articles([article1, article2])

        for story in stories:
            assert isinstance(story, StoryCluster)
            assert story.primary_article is not None, "Each story must have a primary article"
            assert isinstance(story.related_articles, list), "related_articles must be a list"

    def test_story_total_count(self, clusterer):
        """AC2: article_count = 1 primary + len(related)."""
        articles = [
            make_article_dict(
                "India wins cricket match against Australia by 5 wickets",
                "India defeated Australia by 5 wickets at Mumbai.",
                source_slug="ndtv",
            ),
            make_article_dict(
                "India beats Australia in cricket final by five wickets",
                "India beat Australia by five wickets in the final match at Mumbai.",
                source_slug="timesofindia",
            ),
        ]

        stories = clusterer.cluster_articles(articles)
        for story in stories:
            assert story.article_count == 1 + len(story.related_articles)


# ---------------------------------------------------------------------------
# AC3 - Primary article selected by quality score
# ---------------------------------------------------------------------------


class TestQualityScorePrimarySelection:
    def test_quality_score_computed(self, clusterer):
        """AC3: Quality score (source priority + length + image) determines primary article."""
        short_low = make_article_dict(
            title="India wins cricket match",
            body="India won the game.",  # Very short - low quality
            source_slug="zeenews",  # Low priority source
            image_url=None,
        )
        long_high = make_article_dict(
            title="India beats Australia in thrilling cricket match by 5 wickets",
            body="India beat Australia by 5 wickets in a thrilling cricket match. " * 50,  # Long body
            source_slug="thehindu",  # High priority source
            image_url="https://thehindu.com/image.jpg",
        )

        score_short = clusterer.compute_quality_score(short_low)
        score_long = clusterer.compute_quality_score(long_high)
        assert score_long > score_short, "Higher priority source + longer body + image should score higher"

    def test_highest_quality_becomes_primary(self, clusterer):
        """AC3: The article with the highest quality score is the primary in a cluster."""
        low_quality = make_article_dict(
            title="India wins cricket match against Australia by 5 wickets",
            body="India won the game.",
            source_slug="zeenews",  # priority 65
            image_url=None,
        )
        high_quality = make_article_dict(
            title="India beats Australia by five wickets in thrilling match",
            body="India beat Australia by five wickets in a thrilling cricket match at Mumbai. " * 30,
            source_slug="thehindu",  # priority 100
            image_url="https://thehindu.com/photo.jpg",
        )

        stories = clusterer.cluster_articles([low_quality, high_quality])

        assert len(stories) == 1
        primary = stories[0].primary_article
        assert primary["source_slug"] == "thehindu", "The Hindu should be selected as primary due to higher quality"


# ---------------------------------------------------------------------------
# AC4 - New articles checked against existing stories before creating new ones
# ---------------------------------------------------------------------------


class TestStoryMerging:
    def test_new_article_merged_into_existing_story(self, clusterer):
        """AC4: New article similar to existing story primary is merged, not creating a new story."""
        # First batch: creates initial stories
        initial_article = make_article_dict(
            title="India wins cricket match against Australia by 5 wickets",
            body="India defeated Australia by 5 wickets in a thrilling final at Mumbai stadium.",
            source_slug="ndtv",
        )
        initial_stories = clusterer.cluster_articles([initial_article])
        assert len(initial_stories) == 1

        # Second batch: new similar article should merge into existing story
        new_article = make_article_dict(
            title="India beats Australia in cricket match by five wickets",
            body="India beat Australia by five wickets in the final at Mumbai.",
            source_slug="timesofindia",
        )
        updated_stories = clusterer.cluster_articles([new_article], existing_stories=initial_stories)

        assert len(updated_stories) == 1, "Should still have 1 story after merging"
        assert updated_stories[0].article_count == 2, "Story should have 2 articles after merge"

    def test_unrelated_article_creates_new_story(self, clusterer):
        """AC4: Unrelated article does NOT merge into existing story and creates a new one."""
        initial_article = make_article_dict(
            title="India wins cricket match against Australia by 5 wickets",
            body="India defeated Australia by 5 wickets in a thrilling cricket final.",
            source_slug="ndtv",
        )
        initial_stories = clusterer.cluster_articles([initial_article])

        # Completely unrelated article
        unrelated = make_article_dict(
            title="Finance Minister presents Union Budget with tax cuts",
            body="Government reduces income tax for middle class in a landmark budget announcement.",
            source_slug="mint",
        )
        updated_stories = clusterer.cluster_articles([unrelated], existing_stories=initial_stories)

        assert len(updated_stories) == 2, "Unrelated article should create a second story"


# ---------------------------------------------------------------------------
# AC5 - Singleton articles become standalone stories
# ---------------------------------------------------------------------------


class TestSingletonStories:
    def test_single_article_becomes_standalone_story(self, clusterer):
        """AC5: A single article with no similar peers becomes its own standalone story."""
        articles = [
            make_article_dict("ISRO launches new rocket to Mars", "India launched its first Mars mission today."),
            make_article_dict("Budget 2026 presented by Finance Minister", "Income tax rate is reduced."),
        ]
        stories = clusterer.cluster_articles(articles)

        assert len(stories) == 2, "Both articles should be standalone stories"
        for story in stories:
            assert story.primary_article is not None
            assert len(story.related_articles) == 0, "Singleton stories have no related articles"

    def test_mixed_cluster_and_singleton(self, clusterer):
        """AC5: Mix of clusterable and singleton articles processed correctly."""
        arts = [
            make_article_dict(
                "India wins cricket match against Australia",
                "India defeated Australia by 5 wickets at Mumbai.",
                source_slug="ndtv",
            ),
            make_article_dict(
                "India beats Australia in thrilling cricket final",
                "India beat Australia by five wickets in the thrilling final at Mumbai.",
                source_slug="timesofindia",
            ),
            make_article_dict(
                "RBI cuts repo rate by 25 basis points",
                "Central bank of India cuts interest rate to boost economic growth.",
                source_slug="mint",
            ),
        ]
        stories = clusterer.cluster_articles(arts)

        story_counts = sorted([s.article_count for s in stories], reverse=True)
        assert story_counts[0] == 2, "Cricket cluster should have 2 articles"
        assert story_counts[1] == 1, "RBI singleton should have 1 article"


# ---------------------------------------------------------------------------
# AC6 - GET /api/stories/{id} returns story with all related articles
# ---------------------------------------------------------------------------


class TestStoriesAPIEndpoint:
    def _setup_story(self, db_session, engine):
        """Create source, category, articles, and story in test DB."""
        cat = Category(id=50, slug="sports", name="Sports", is_active=True)
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

        story_id = uuid.uuid4()
        story = Story(id=story_id)
        db_session.add(story)
        db_session.flush()

        primary = make_article_orm(
            db_session,
            src,
            cat,
            title="India wins cricket match",
            body="India defeated Australia in a thrilling match.",
        )
        primary.story_id = story_id

        related = make_article_orm(
            db_session,
            src,
            cat,
            title="India beats Australia in cricket",
            body="India beat Australia in cricket.",
        )
        related.story_id = story_id

        story.primary_article_id = primary.id
        db_session.commit()

        return story_id, primary.id, related.id

    def test_get_story_by_id(self, db_session, api_client, engine):
        """AC6: GET /api/stories/{id} returns correct story with primary and related articles."""
        story_id, primary_id, related_id = self._setup_story(db_session, engine)

        response = api_client.get(f"/api/stories/{story_id}")
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == str(story_id)
        assert data["primary_article"] is not None
        assert data["primary_article"]["id"] == str(primary_id)
        assert data["article_count"] >= 1
        assert isinstance(data["related_articles"], list)

    def test_get_story_not_found(self, api_client):
        """AC6: Returns 404 for non-existent story ID."""
        fake_id = str(uuid.uuid4())
        response = api_client.get(f"/api/stories/{fake_id}")
        assert response.status_code == 404

    def test_get_story_invalid_id(self, api_client):
        """AC6: Returns 400 for invalid UUID format."""
        response = api_client.get("/api/stories/not-a-uuid")
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# AC7 - Clustering runs in < 10 seconds for 500 articles
# ---------------------------------------------------------------------------


class TestClusteringPerformance:
    def test_500_articles_under_10_seconds(self, clusterer):
        """AC7: Clustering 500 articles completes in < 10 seconds."""
        topics = [
            ("India wins cricket match", "India defeated Australia in the final match at Mumbai."),
            ("Budget 2026 highlights", "Finance Minister announces income tax cuts and reforms."),
            ("ISRO launches satellite", "Indian Space Research Organisation successfully put satellite in orbit."),
            ("Monsoon arrives in Kerala", "Southwest monsoon has officially made landfall in Kerala."),
            ("RBI cuts interest rates", "Reserve Bank reduced repo rate by 25 basis points today."),
        ]

        articles = []
        for i in range(500):
            title, body = topics[i % len(topics)]
            articles.append(
                make_article_dict(
                    title=f"{title} — report #{i}",
                    body=f"{body} Additional context for article number {i}.",
                    source_slug="ndtv" if i % 2 == 0 else "timesofindia",
                )
            )

        start = time.time()
        stories = clusterer.cluster_articles(articles)
        elapsed = time.time() - start

        assert len(stories) > 0
        assert len(stories) <= len(articles)
        assert elapsed < 10.0, f"Clustering 500 articles took {elapsed:.2f}s, limit is 10.0s"
