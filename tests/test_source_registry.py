import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.scrapers.source_registry import SourceConfig, SourceRegistry

# Use in-memory SQLite for testing
engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module")
def db_session():
    from src.models.source import Source

    Source.__table__.create(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Source.__table__.drop(bind=engine)


@pytest.fixture
def registry():
    return SourceRegistry()


def test_load_from_disk(registry):
    loaded_count = registry.load_from_disk()
    assert loaded_count >= 10, "Should have loaded at least 10 source configs"

    # Check if a specific config is loaded correctly
    ndtv = registry.get_config("ndtv")
    assert ndtv is not None
    assert ndtv.name == "NDTV"
    assert ndtv.language == "en"
    assert ndtv.scrape_type == "playwright"


def test_sync_with_db(registry, db_session):
    registry.load_from_disk()
    registry.sync_with_db(db_session)

    sources = registry.get_all_sources(db_session)
    assert len(sources) >= 10

    ndtv_source = next((s for s in sources if s.slug == "ndtv"), None)
    assert ndtv_source is not None
    assert ndtv_source.name == "NDTV"
    assert ndtv_source.is_active is True


def test_invalid_config():
    with pytest.raises(ValidationError):
        SourceConfig(
            name="Invalid",
            # Missing required fields like base_url, article_list_url
        )


def test_update_health(registry, db_session):
    # Ensure it is in DB
    registry.load_from_disk()
    registry.sync_with_db(db_session)

    # Test success
    source = registry.update_health(db_session, "ndtv", success=True)
    assert source is not None
    assert source.success_count == 1
    assert source.error_count == 0
    assert source.is_healthy is True

    # Test error
    source = registry.update_health(db_session, "ndtv", success=False)
    assert source.error_count == 1
