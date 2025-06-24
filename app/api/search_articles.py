from sqlalchemy.sql import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Article
import app.models.utils as model_utils


async def get_articles_by_tag_names(
    db_engine: AsyncEngine,
    tag_names: str,
    limit: int = 10,
):
    """
    Retrieve articles based on a set of provided tag names and rank them using an embedding similarity approach.
    The function combines articles associated with specified tags and those deemed contextually similar based on
    an embedding generated from the tag names. Results include the most relevant articles ordered by similarity
    to the generated embedding vector or tag associations.

    :param db_engine: An asynchronous SQLAlchemy engine used to execute database queries.
    :param tag_names: A string containing comma-separated tag names for filtering articles.
    :param limit: An optional integer defining the maximum number of articles to return. Defaults to 10.
    :return: A list of dictionaries, where each dictionary represents an article with attributes such as summarization,
            distance, link, headline, published date, associated tag names, and key insights. The list is sorted based
            on cardinality of associated tags and embedding similarity.
    """
    tags = tag_names.split(",")
    query_vec = model_utils.generate_embedding("\n ".join(tags))

    select_stmt = """
    WITH cte_tags AS (
        SELECT
            a.article_id AS article_id,
            ARRAY_AGG(DISTINCT t.tag) AS tag_names
        FROM tags t 
        JOIN article_tags a
            ON t.tag_id = a.tag_id
        WHERE t.tag = ANY(:tags)
        GROUP BY 1
    ),
    cte_embedding AS (
        SELECT
            article_id,
            summarization,
            embedding <=> (:query_vec)::vector AS distance
        FROM embeddings
        ORDER BY embedding <=> (:query_vec)::vector
        LIMIT :limit
    ),
    cte_embeddings_data AS (
        SELECT
            ROW_NUMBER() OVER (ORDER BY e.distance) as embedding_position_id,
            e.article_id AS article_id,
            e.summarization AS summarization,
            e.distance AS distance,
            a.link AS link,
            a.link_hash AS link_hash,
            a.headline AS headline,
            a.date AS published_date,
            COALESCE(t.tag_names, ARRAY[]::text[]) AS tag_names,
            ARRAY_AGG(k.key_insight) AS key_insights
        FROM cte_embedding AS e
        JOIN articles AS a
            ON a.article_id = e.article_id
        JOIN key_insights AS k
            On a.article_id = k.article_id
        LEFT JOIN cte_tags AS t
            ON a.article_id = t.article_id
        GROUP BY 2, 3, 4, 5, 6, 7, 8, 9
    ),
    cte_tags_limited AS (
        SELECT
            article_id,
            tag_names
        FROM cte_tags
        ORDER BY cardinality(tag_names) DESC
        LIMIT :limit
    ),
    cte_articles_by_tags_data AS (
        SELECT
            t.article_id,
            e.summarization AS summarization,
            e.embedding <=> (:query_vec)::vector AS distance,
            a.link AS link,
            a.link_hash AS link_hash,
            a.headline AS headline,
            a.date AS published_date,
            t.tag_names AS tag_names,
            ARRAY_AGG(k.key_insight) AS key_insights
        FROM cte_tags_limited AS t
        JOIN embeddings AS e
            ON t.article_id = e.article_id
        JOIN articles AS a
            ON e.article_id = a.article_id
        JOIN key_insights AS k
            ON t.article_id = k.article_id
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
    )
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY
            cardinality(COALESCE(a.tag_names, e.tag_names)) DESC,
            COALESCE(a.distance, e.distance)
        ) AS article_position_id,
        COALESCE(e.embedding_position_id, NULL) AS embedding_position_id,
        COALESCE(a.summarization, e.summarization) AS summarization,
        ROUND(COALESCE(a.distance, e.distance)::numeric, 4) AS distance,
        COALESCE(a.link, e.link) AS link,
        COALESCE(a.link_hash, e.link_hash) AS link_hash,
        COALESCE(a.headline, e.headline) AS headline,
        COALESCE(a.published_date, e.published_date) AS published_date,
        COALESCE(a.tag_names, e.tag_names) AS tag_names,
        COALESCE(a.key_insights, e.key_insights) AS key_insights
    FROM cte_articles_by_tags_data AS a
    FULL OUTER JOIN cte_embeddings_data AS e
        ON a.article_id = e.article_id
    """

    async with db_engine.connect() as conn:
        result = await conn.execute(text(select_stmt), {
            "tags": tags,
            "limit": limit,
            "query_vec": str(query_vec),
        })
        rows = result.fetchall()
        return [row._asdict() for row in rows] if rows else []


async def get_articles_by_link_hash(
    db_session: AsyncSession,
    link_hash: str
):
    """
    Retrieves article details from the database by link hash. Queries the database
    using the provided asynchronous session to fetch an article matching the
    specified link hash. If found, the details of the article, along with its
    associated embedding, tags, and key insights, are extracted and structured
    as a dictionary. Returns `None` if no matching article is found.

    :param db_session: An instance of AsyncSession used to interact with the database in an asynchronous manner.
    :param link_hash: The unique hash of the link for which the article details are to be retrieved.
    :return: A dictionary containing article details including link, link hash, headline, published date,
             summarization, tag names, and key insights, or `None` if the article is not found.
    """
    async with db_session as session:
        stmt = select(Article).options(
            selectinload(Article.embedding),
            selectinload(Article.tags),
            selectinload(Article.key_insights)
        ).where(
            Article.link_hash == link_hash
        )
        result = await session.execute(stmt)
        article = result.scalar_one_or_none()
        output = None
        if article:
            output = {
                "link": article.link,
                "link_hash": article.link_hash,
                "headline": article.headline,
                "published_date": article.date,
                "summarization": article.embedding.summarization,
                "tag_names": [row.tag for row in article.tags],
                "key_insights": [row.key_insight for row in article.key_insights]
            }
        return output
