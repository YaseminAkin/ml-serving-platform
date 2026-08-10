"""Shared pytest fixtures for DB-backed integration tests."""

import os

import pytest
import redis as redis_sync
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from db.models import Base
from gateway.main import app

# Alembic/app use the async asyncpg driver; fixture setup here is plain
# sync SQLAlchemy so it never touches the app's event loop.
TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://mlserving:mlserving@localhost:5432/mlserving",
).replace("+asyncpg", "+psycopg2")

TEST_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


@pytest.fixture(scope="session")
def sync_engine():
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def _clean_tables(sync_engine):
    with sync_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture(autouse=True)
def _clean_redis():
    client = redis_sync.Redis.from_url(TEST_REDIS_URL)
    client.flushdb()
    client.close()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
