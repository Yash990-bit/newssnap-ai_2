import os
from unittest import mock

from src.config.settings import Settings


def test_settings_docker_url_adjustment():
    """Test that Docker URLs are adjusted to localhost when running outside Docker."""
    with mock.patch("os.path.exists", return_value=False), mock.patch.dict(os.environ, {"DOCKER_CONTAINER": ""}):
        settings = Settings(DATABASE_URL="postgresql://user:pass@db:5432/dbname", REDIS_URL="redis://redis:6379/0")
        assert "localhost" in settings.DATABASE_URL
        assert "db:5432" not in settings.DATABASE_URL
        assert "localhost" in settings.REDIS_URL
        assert "redis:6379" not in settings.REDIS_URL


def test_settings_docker_url_no_adjustment():
    """Test that URLs are NOT adjusted when running inside Docker."""
    with mock.patch("os.path.exists", return_value=True):
        settings = Settings(DATABASE_URL="postgresql://user:pass@db:5432/dbname", REDIS_URL="redis://redis:6379/0")
        assert "db" in settings.DATABASE_URL
        assert "redis" in settings.REDIS_URL
