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


@pytest.fixture(scope="function")
def science_category_extended_data():
    return [
        {
            "link": 'https://www.huffpost.com/entry/cat-on-the-mat',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/cat-on-the-mat'),
            "headline": "A cat on the mat: The story of a cat who lost his home",
            "date": "2024-01-03"
        },
        {
            "link": 'https://www.huffpost.com/entry/cat-saved-by-firefighter',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/cat-saved-by-firefighter'),
            "headline": "A cat saved by a firefigher from a burning building",
            "date": "2024-01-04"
        },
        {
            "link": 'https://www.huffpost.com/entry/dog-rescue-shelter',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/dog-rescue-shelter'),
            "headline": "Local shelter saves 50 dogs from terrible conditions",
            "date": "2024-01-05"
        },
        {
            "link": 'https://www.huffpost.com/entry/elephant-sanctuary',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/elephant-sanctuary'),
            "headline": "New elephant sanctuary opens in Thailand",
            "date": "2024-01-06"
        },
        {
            "link": 'https://www.huffpost.com/entry/panda-twins',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/panda-twins'),
            "headline": "Rare panda twins born in Singapore Zoo",
            "date": "2024-01-07"
        },
        {
            "link": 'https://www.huffpost.com/entry/penguin-rescue',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/penguin-rescue'),
            "headline": "Rescue team saves stranded penguins in Antarctica",
            "date": "2024-01-08"
        },
        {
            "link": 'https://www.huffpost.com/entry/lion-cubs',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/lion-cubs'),
            "headline": "Three lion cubs make their debut at national zoo",
            "date": "2024-01-09"
        },
        {
            "link": 'https://www.huffpost.com/entry/dolphin-intelligence',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/dolphin-intelligence'),
            "headline": "New study reveals amazing dolphin communication abilities",
            "date": "2024-01-10"
        },
        {
            "link": 'https://www.huffpost.com/entry/gorilla-sign-language',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/gorilla-sign-language'),
            "headline": "Gorilla learns sign language: Breakthrough in primate studies",
            "date": "2024-01-11"
        },
        {
            "link": 'https://www.huffpost.com/entry/koala-recovery',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/koala-recovery'),
            "headline": "Koala population shows signs of recovery after bushfires",
            "date": "2024-01-12"
        },
        {
            "link": 'https://www.huffpost.com/entry/tiger-conservation',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/tiger-conservation'),
            "headline": "Tiger numbers rise in Indian wildlife reserve",
            "date": "2024-01-13"
        },
        {
            "link": 'https://www.huffpost.com/entry/polar-bear-arctic',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/polar-bear-arctic'),
            "headline": "Scientists track polar bear migration patterns",
            "date": "2024-01-14"
        },
        {
            "link": 'https://www.huffpost.com/entry/new-iphone-release',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/new-iphone-release'),
            "headline": "Apple announces revolutionary features in latest iPhone",
            "date": "2024-01-15"
        },
        {
            "link": 'https://www.huffpost.com/entry/android-update',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/android-update'),
            "headline": "Major Android update brings AI features to millions of phones",
            "date": "2024-01-16"
        },
        {
            "link": 'https://www.huffpost.com/entry/phone-battery-breakthrough',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/phone-battery-breakthrough'),
            "headline": "New battery technology promises week-long phone charge",
            "date": "2024-01-17"
        },
        {
            "link": 'https://www.huffpost.com/entry/foldable-phones-future',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/foldable-phones-future'),
            "headline": "Foldable phones reshape the future of mobile technology",
            "date": "2024-01-18"
        },
        {
            "link": 'https://www.huffpost.com/entry/5g-network-expansion',
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/5g-network-expansion'),
            "headline": "Global 5G network expansion accelerates smartphone capabilities",
            "date": "2024-01-19"
        }
    ]


@pytest.fixture(scope="function")
def llm_summaries_extended_data():
    return [
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/cat-on-the-mat'),
            "summarization": "A heartbreaking story about a homeless cat found in a local park. The community rallied together to help find the cat a new home.",
            "tags": ["homeless-cat", "community", "rescue"],
            "key_insights": ["A homeless cat was discovered in a local park.",
                             "The community worked together to help the cat."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/cat-saved-by-firefighter'),
            "summarization": "A brave firefighter rescued a cat from a burning building. The cat was safely returned to its grateful owner after receiving medical attention.",
            "tags": ["cat-rescue", "firefighter", "hero"],
            "key_insights": ["A firefighter risked their life to save a cat from a fire.",
                             "The cat was successfully reunited with its owner."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/dog-rescue-shelter'),
            "summarization": "A local animal shelter conducted a major rescue operation saving 50 dogs from poor living conditions. The dogs are now receiving proper care and rehabilitation.",
            "tags": ["dog-rescue", "shelter", "animal-welfare"],
            "key_insights": ["50 dogs were rescued from poor living conditions.",
                             "The shelter is providing rehabilitation and care for the rescued dogs."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/elephant-sanctuary'),
            "summarization": "A new sanctuary for elephants has opened in Thailand, providing a safe haven for rescued and retired elephants. The facility focuses on natural habitat and ethical treatment.",
            "tags": ["elephant", "sanctuary", "animal-welfare"],
            "key_insights": ["New elephant sanctuary opens to provide safe haven in Thailand.",
                             "The sanctuary emphasizes natural habitat and ethical treatment."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/panda-twins'),
            "summarization": "Singapore Zoo celebrates the rare birth of panda twins. The cubs are healthy and being carefully monitored by veterinary staff.",
            "tags": ["panda", "zoo", "wildlife"],
            "key_insights": ["Rare panda twins were born at Singapore Zoo.",
                             "Both cubs are healthy and under professional care."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/penguin-rescue'),
            "summarization": "A rescue team in Antarctica successfully saved a group of stranded penguins. The operation involved careful planning and coordination in extreme weather conditions.",
            "tags": ["penguin", "rescue", "antarctica"],
            "key_insights": ["Rescue team saved stranded penguins in Antarctica.",
                             "The operation was conducted under challenging weather conditions."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/lion-cubs'),
            "summarization": "Three lion cubs made their first public appearance at the national zoo. The cubs represent a significant success for the zoo's breeding program.",
            "tags": ["lion-cubs", "zoo", "wildlife"],
            "key_insights": ["Three lion cubs debuted at the national zoo.",
                             "The births mark a success for the zoo's breeding program."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/dolphin-intelligence'),
            "summarization": "New research reveals complex communication patterns among dolphins. Scientists discovered previously unknown aspects of dolphin language and social interaction.",
            "tags": ["dolphin", "research", "communication"],
            "key_insights": ["Study reveals new aspects of dolphin communication.",
                             "Findings show complex social interaction patterns among dolphins."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/gorilla-sign-language'),
            "summarization": "Scientists achieve breakthrough as gorilla masters sign language communication. This development provides new insights into primate cognitive abilities.",
            "tags": ["gorilla", "sign-language", "research"],
            "key_insights": ["Gorilla successfully learned to communicate through sign language.",
                             "The achievement provides new understanding of primate intelligence."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/koala-recovery'),
            "summarization": "Australian koala populations show positive recovery trends following devastating bushfires. Conservation efforts and habitat restoration programs prove successful.",
            "tags": ["koala", "conservation", "recovery"],
            "key_insights": ["Koala populations are recovering after bushfire damage.",
                             "Conservation efforts have helped support population growth."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/tiger-conservation'),
            "summarization": "Indian wildlife reserve reports significant increase in tiger population. Conservation efforts and anti-poaching measures contribute to the success.",
            "tags": ["tiger", "conservation", "wildlife"],
            "key_insights": ["Tiger numbers have increased in Indian wildlife reserve.",
                             "Success attributed to conservation and anti-poaching efforts."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/polar-bear-arctic'),
            "summarization": "Scientists conduct comprehensive study of polar bear migration patterns. Research provides crucial data for arctic wildlife conservation efforts.",
            "tags": ["polar-bear", "research", "arctic"],
            "key_insights": ["Scientists tracked polar bear migration patterns.",
                             "Research data will help inform conservation strategies."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/new-iphone-release'),
            "summarization": "Apple announces groundbreaking features in their latest iPhone model. The new device includes advanced AI capabilities and improved battery life.",
            "tags": ["iphone", "technology", "innovation"],
            "key_insights": ["New iPhone introduces revolutionary features.",
                             "Device includes AI capabilities and better battery performance."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/android-update'),
            "summarization": "Major Android system update brings AI-powered features to smartphones. The update will be available to millions of users worldwide.",
            "tags": ["android", "ai-features", "technology"],
            "key_insights": ["Android releases major update with AI features.",
                             "Update will reach millions of users globally."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/phone-battery-breakthrough'),
            "summarization": "Scientists develop revolutionary battery technology enabling week-long phone charge. The breakthrough could transform mobile device usage patterns.",
            "tags": ["battery-tech", "innovation", "mobile"],
            "key_insights": ["New battery technology enables week-long phone charge.",
                             "Innovation could change how people use mobile devices."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/foldable-phones-future'),
            "summarization": "Foldable phones are revolutionizing mobile technology industry. New designs and improvements in durability drive increased adoption.",
            "tags": ["foldable-phones", "technology", "innovation"],
            "key_insights": ["Foldable phones are changing mobile technology landscape.",
                             "Improved designs lead to wider adoption."]
        },
        {
            "link_hash": make_hashed_url('https://www.huffpost.com/entry/5g-network-expansion'),
            "summarization": "Global 5G network expansion enables enhanced smartphone capabilities. The technology promises faster speeds and new mobile applications.",
            "tags": ["5g-network", "technology", "mobile"],
            "key_insights": ["5G network expansion enhances smartphone capabilities.",
                             "Technology enables faster speeds and new applications."]
        }
    ]


@pytest.fixture(scope="function")
def science_category_extended_file(science_category_extended_data):
    data = '\n'.join(json.dumps(item) for item in science_category_extended_data)
    file = BytesIO(data.encode())
    file.name = "science_category_extended.jsonl"
    return file


@pytest.fixture(scope="function")
def science_category_llm_summaries_extended_file(llm_summaries_extended_data):
    data = '\n'.join(json.dumps(item) for item in llm_summaries_extended_data)
    file = BytesIO(data.encode())
    file.name = "science_category_llm_summaries_extended_file.jsonl"
    return file
