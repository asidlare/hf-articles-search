import pandas as pd
import os
from datetime import date

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import exists
from tqdm import tqdm
from app.services.database import get_db_session
from app.models import Article, Tag, ArticleTag, KeyInsight, Embedding


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
CATEGORY_FILE_PATH = os.path.join(ROOT_PATH, "science_category.jsonl")
SUMMARY_FILE_PATH = os.path.join(ROOT_PATH, "science_category_llm_summaries.jsonl")


async def add_article(
    session: AsyncSession,
    link_hash: str,
    link: str,
    headline: str,
    publication_date: date
) -> Article:
    """
    Adds a new Article record to the database. This function is designed to create
    and add an Article object to the provided session, using the given details.

    :param session: Database session used for interaction with the database.
    :type session: AsyncSession
    :param link_hash: Unique hash representing the link of the article.
    :type link_hash: str
    :param link: URL to the article.
    :type link: str
    :param headline: The headline or title of the article.
    :type headline: str
    :param publication_date: The publication date of the article.
    :type publication_date: date
    :return: The created Article object for the newly added article.
    :rtype: Article
    """
    article = Article(
        link_hash=link_hash,
        link=link,
        headline=headline,
        date=publication_date,
    )
    session.add(article)
    await session.flush()
    return article


async def add_embedding(
    session: AsyncSession,
    article: Article,
    summarization: str
) -> None:
    """
    Adds an embedding record to the database with the provided article and
    summarization. This function creates an instance of the Embedding
    model, associates it with the given article and summarization, and
    adds it to the session. The changes are prepared for persistence
    by flushing the session.

    :param session: The asynchronous database session used to add and
                    manage the embedding record.
    :type session: AsyncSession
    :param article: The article object associated with the embedding.
    :type article: Article
    :param summarization: The summarization text to be stored as part of the
                          embedding.
    :type summarization: str
    :return: None
    """
    embedding = Embedding(article=article,summarization=summarization)
    session.add(embedding)
    await session.flush()


async def add_tags(
    session: AsyncSession,
    article: Article,
    tags: list[str]
) -> None:
    """
    Adds tags to an article in the database with the provided session, ensuring
    existent tags are reused and new ones are created as necessary. Also creates
    associations between the article and the tags in the database.

    :param session: The asynchronous database session used to interact with the
        database.
    :type session: AsyncSession
    :param article: The article instance to which the tags will be associated.
    :type article: Article
    :param tags: A list of tag names to associate with the specified article. Non-
        existent tags will be created.
    :type tags: list[str]
    :return: None
    """
    stmt = select(Tag).where(Tag.tag.in_(tags))
    existing_tags_objects = (await session.scalars(stmt)).all()
    existing_tags = {tag.tag: tag for tag in existing_tags_objects}
    missing_tags = set(tags) - set(existing_tags.keys())
    missing_tags_objects = []
    article_tag_relations = []

    for tag in missing_tags:
        missing_tags_objects.append(Tag(tag=tag))
    session.add_all(missing_tags_objects)
    await session.flush()

    stmt = select(Tag).where(Tag.tag.in_(tags))
    tags_objects = (await session.scalars(stmt)).all()

    for tag in tags_objects:
        article_tag_relations.append(ArticleTag(article=article, tag=tag))
    session.add_all(article_tag_relations)
    await session.flush()


async def add_key_insights(
    session: AsyncSession,
    article: Article,
    key_insights: list[str]
):
    """
    Asynchronously adds key insights to a given article and persists them in the database
    session. Creates `KeyInsight` objects for each provided key insight and associates
    them with the article. Adds the created objects to the session for database persistence.

    :param session:
        An instance of `AsyncSession` used to interact with the database asynchronously.
    :param article:
        The `Article` object to which the key insights belong.
    :param key_insights:
        A list of key insights to be added to the article. Each item in the list represents
        an individual key insight.
    :return:
        None
    """
    key_insights_objects = []
    for key_insight in key_insights:
        key_insights_objects.append(KeyInsight(article=article, key_insight=key_insight))
    session.add_all(key_insights_objects)
    await session.flush()


async def upload_article(
    db_session: AsyncSession,
    link: str,
    link_hash: str,
    headline: str,
    publication_date: date,
    summarization: str,
    tags: list[str],
    key_insights: list[str]
):
    """
    Uploads an article to the database if it does not already exist. The function first
    checks if the article with the provided ``link_hash`` is already present in the
    database. If the article does not exist, it proceeds to add the article and
    its associated data, including summarization, tags, and key insights. All
    database operations are performed as part of an asynchronous session.

    :param db_session: The asynchronous database session used for executing queries
        and persisting data.
    :type db_session: AsyncSession

    :param link: The original URL of the article to be uploaded.
    :type link: str

    :param link_hash: A unique hash derived from the article's link, serving as a
        primary identifier to check if the article already exists in the database.
    :type link_hash: str

    :param headline: The headline or title of the article.
    :type headline: str

    :param publication_date: The publication date of the article.
    :type publication_date: date

    :param summarization: A summarized version of the article's contents.
    :type summarization: str

    :param tags: A list of strings representing the tags to be associated with
        the article.
    :type tags: list[str]

    :param key_insights: A list of strings encapsulating key insights or important
        takeaways from the article.
    :type key_insights: list[str]

    :return: None if the article already exists in the database. When the article is
        added successfully, no explicit return value is provided by this function.
    :rtype: None
    """
    stmt = select(exists().where(Article.link_hash == link_hash))
    result = await db_session.execute(stmt)
    exists_flag = result.scalar_one()
    if exists_flag:
        return

    async with db_session as session:
        article = await add_article(
            session=session,
            link_hash=link_hash,
            link=link,
            headline=headline,
            publication_date=publication_date
        )
        await add_embedding(
            session=session,
            article=article,
            summarization=summarization,
        )
        await add_tags(
            session=session,
            article=article,
            tags=tags,
        )
        await add_key_insights(
            session=session,
            article=article,
            key_insights=key_insights,
        )
        await session.commit()


async def upload_articles(
    category_file: str = CATEGORY_FILE_PATH,
    summary_file: str = SUMMARY_FILE_PATH,
) -> None:
    """
    Asynchronously uploads articles by merging category and summary data, and then
    iterating through the merged dataset to process and upload each article.

    The function reads in data from specified category and summary JSON lines
    files, merges the datasets on the field `link_hash`, and processes important
    fields such as `link`, `headline`, `date`, `summarization`, `tags`, and
    `key_insights`. Each row of the merged dataset is uploaded individually
    through a database session.

    :param category_file: Path to the JSON lines file containing category data.
        Defaults to CATEGORY_FILE_PATH.
    :param summary_file: Path to the JSON lines file containing summary data.
        Defaults to SUMMARY_FILE_PATH.
    :return: This function does not return any value. It performs side-effects by
        uploading data to a database asynchronously.
    """
    category_df = pd.read_json(category_file, lines=True)
    summary_df = pd.read_json(summary_file, lines=True)
    data_df = pd.merge(
        category_df,
        summary_df,
        on="link_hash"
    )[[
        "link",
        "link_hash",
        "headline",
        "date",
        "summarization",
        "tags",
        "key_insights"
    ]]
    total_rows = len(data_df)

    # Add tqdm progress bar for iteration
    for _, row in tqdm(data_df.iterrows(), total=total_rows, desc="Processing articles"):
        # get db_session
        db_session = await anext(get_db_session())
        # process article
        await upload_article(
            db_session=db_session,
            link=row["link"],
            link_hash=row["link_hash"],
            headline=row["headline"],
            publication_date=row["date"].date(),
            summarization=row["summarization"],
            tags=row["tags"],
            key_insights=row["key_insights"],
        )


async def get_articles_by_link_hash(link_hash: str) -> Article:
    """
    Retrieve an article from the database based on the provided link hash.

    This function asynchronously queries the database using the provided link
    hash to fetch an `Article` object. The function utilizes SQLAlchemy to perform
    the database operation and ensures that the related attributes such as `embedding`,
    `tags`, and `key_insights` of the `Article` model are loaded as well.

    :param link_hash: The hash of the link used to identify a particular article.
    :type link_hash: str
    :return: The `Article` object corresponding to the provided link hash or None
             if no matching article is found.
    :rtype: Article
    """
    db_session = await anext(get_db_session())
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
        return article
