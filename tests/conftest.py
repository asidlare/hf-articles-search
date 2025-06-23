import json
import numpy as np
import pytest
import torch
from contextlib import ExitStack
from fastapi.testclient import TestClient
from io import BytesIO
from pytest_asyncio import is_async_test
from pytest_postgresql import factories
from pytest_postgresql.janitor import DatabaseJanitor
from sentence_transformers import SentenceTransformer
from sqlalchemy.sql import text


from app.main import init_app
from app.services.database import (
    get_db_engine,
    get_db_session,
    databasemanager
)
from app.models.utils import generate_embedding
from app.transformations.scrapper import make_hashed_url


def pytest_collection_modifyitems(items):
    pytest_asyncio_tests = (item for item in items if is_async_test(item))
    session_scope_marker = pytest.mark.asyncio(loop_scope="session")
    for async_test in pytest_asyncio_tests:
        async_test.add_marker(session_scope_marker, append=False)


@pytest.fixture(autouse=True)
def app():
    with ExitStack():
        yield init_app(init_db=False)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


test_db = factories.postgresql_noproc(
    host='db',
    dbname='postgres_test',
    port=5432,
    user='postgres',
    password='postgres'
)


@pytest.fixture(scope="session", autouse=True)
async def connection_test(test_db):
    pg_host = test_db.host
    pg_port = test_db.port
    pg_user = test_db.user
    pg_db = test_db.dbname
    pg_password = test_db.password
    pg_version = test_db.version
    with DatabaseJanitor(
        dbname=pg_db,
        user=pg_user,
        host=pg_host,
        port=pg_port,
        version=pg_version,
        password=pg_password,
    ):
        connection_str = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        databasemanager.init(connection_str)
        yield
        await databasemanager.close()


@pytest.fixture(scope="function", autouse=True)
async def create_tables(connection_test):
    async with databasemanager.connect() as connection:
        await databasemanager.drop_all(connection)
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"));
        await databasemanager.create_all(connection)


@pytest.fixture(scope="function", autouse=True)
async def session_override(app, connection_test):
    async def get_db_session_override():
        async with databasemanager.session() as session:
            yield session

    app.dependency_overrides[get_db_session] = get_db_session_override


@pytest.fixture(scope="function", autouse=True)
async def engine_override(app, connection_test):
    async def get_db_engine_override():
        async with databasemanager.engine() as engine:
            yield engine

    app.dependency_overrides[get_db_engine] = get_db_engine_override


@pytest.fixture(scope="session")
def embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


@pytest.fixture(scope="function")
def generate_embedding(embedding_model, monkeypatch):
    def _generate_embedding(text: str) -> list[float]:
        """Generate embedding using sentence-transformers model"""
        with torch.no_grad():
            embedding = embedding_model.encode(text, convert_to_numpy=True)

        # Upscale to 1536 dimensions by duplicating and adding small random variations
        duplicated = np.repeat(embedding, 2)  # Duplicate each value

        # Use text as seed for reproducible randomness
        # Convert hash to positive number by using absolute value
        seed = abs(hash(text)) % (2 ** 32)  # Ensure it's within numpy's random seed range
        rng = np.random.default_rng(seed=seed)
        noise = rng.normal(0, 0.01, size=1536)  # Small random variations

        # Combine and normalize
        final_embedding = duplicated + noise
        final_embedding = final_embedding / np.linalg.norm(final_embedding)

        return final_embedding.tolist()

    # Import the module where original generate_embedding is defined
    import app.models.utils
    monkeypatch.setattr(app.models.utils, "generate_embedding", _generate_embedding)

    return _generate_embedding


@pytest.fixture(scope="function")
def link_1():
    return 'https://example.com/test-article-1'


@pytest.fixture(scope="function")
def link_1_hash(link_1):
    return make_hashed_url(link_1)


@pytest.fixture(scope="function")
def article_1(link_1, link_1_hash):
    line = {
        "link": link_1,
        "link_hash": link_1_hash,
        "headline": "First Article Title",
        "date": "2024-01-02"
    }
    return json.dumps(line)


@pytest.fixture(scope="function")
def link_2():
    return 'https://example.com/test-article-2'


@pytest.fixture(scope="function")
def link_2_hash(link_2):
    return make_hashed_url(link_2)


@pytest.fixture(scope="function")
def article_2(link_2, link_2_hash):
    line = {
        "link": link_2,
        "link_hash": link_2_hash,
        "headline": "Second Article Title",
        "date": "2024-01-03"
    }
    return json.dumps(line)


@pytest.fixture(scope="function")
def llm_summary_1(link_1):
    link_hashed = make_hashed_url(link_1)
    line = {
        "link_hash": link_hashed,
        "summarization": "It is a summary of the first article.",
        "tags": ["tag1", "new-tag", "tag2"],
        "key_insights": ["First insight.", "Second insight."]
    }
    return json.dumps(line)


@pytest.fixture(scope="function")
def llm_summary_2(link_2):
    link_hashed = make_hashed_url(link_2)
    line = {
        "link_hash": link_hashed,
        "summarization": "It is a summary of the second article.",
        "tags": ["tag1", "next-tag", "tag3"],
        "key_insights": ["Third insight.", "Fourth insight."]
    }
    return json.dumps(line)


@pytest.fixture(scope="function")
def science_category_file(article_1, article_2):
    data = article_1 + '\n' + article_2
    file = BytesIO(data.encode())
    file.name = "science_category.jsonl"
    return file


@pytest.fixture(scope="function")
def science_category_llm_summaries_file(llm_summary_1, llm_summary_2):
    data = llm_summary_1 + '\n' + llm_summary_2
    file = BytesIO(data.encode())
    file.name = "science_category_llm_summaries.jsonl"
    return file
