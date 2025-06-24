from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from pydantic import ValidationError
from typing import Annotated

from app.api.search_articles import get_articles_by_tag_names, get_articles_by_link_hash
from app.services.database import get_db_session, get_db_engine
from app.schemas.search_articles import ArticleResponse, TagSearchResponse, Link, LinkResponse
from app.transformations.scrapper import make_hashed_url


search_articles_router = APIRouter()


@search_articles_router.get(
    "/search-articles-by-tag-names",
    status_code=status.HTTP_200_OK,
    response_model=TagSearchResponse,
    operation_id="search_articles_by_tag_names",
)
async def search_articles_by_tag_names(
    db_engine: Annotated[AsyncEngine, Depends(get_db_engine)],
    tag_names: str = Query(..., description="A comma-separated list of tag names to search for."),
    limit: int = Query(10, description="The limit applied separately to results fetched by tags and by cosine distance")
):
    """
    Handles the endpoint for searching articles by tag names. The function takes a
    list of tags to search for and a limit to restrict the number of results for
    tags and cosine distance. It interacts with the database to fetch articles
    matching the specified tags and formats the response.

    :param db_engine: An instance of AsyncEngine (as a dependency) which provides
        the database connection needed for querying the articles.
    :param tag_names: A comma-separated list of tag names to search for.
    :param limit: The maximum number of articles to retrieve for search results
        based on both tag matching and cosine similarity. Default is 10 for each separately.
    :return: A `TagSearchResponse`, encapsulating the list of articles that match
        the search criteria.
    :raises: Returns a JSONResponse with a 400 status code and an error message if
        `tag_names` is not provided in the query parameters.
    """
    if not tag_names:
        return JSONResponse(
            content={"message": "Bad request: tag_names is required"},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    results = await get_articles_by_tag_names(db_engine, tag_names)
    return TagSearchResponse(articles=results)


@search_articles_router.get(
    "/get-article-by-link-hash",
    status_code=status.HTTP_200_OK,
    response_model=ArticleResponse,
    operation_id="get_article_by_link_hash",
)
async def get_article_by_link_hash(
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    link_hash: str = Query(..., description="An url link hash to search for.")
):
    """
    Retrieve an article by its link hash.

    This function is an API endpoint that fetches an article from the database
    based on the provided link hash. It utilizes the given database session to
    query the link hash and returns the corresponding article details in the
    response model.

    :param db_session: An asynchronous database session dependency injected for
        performing database operations.
    :param link_hash: A unique hash string representing a link, which is used
        to identify the article in the database.
    :return: An instance of ArticleResponse containing the details of the requested article.
    """
    if not link_hash:
        return JSONResponse(
            content={"message": "Bad request: link_hash is required"},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    results = await get_articles_by_link_hash(db_session, link_hash)
    if not results:
        return JSONResponse(
            content={"message": f"Article link hash {link_hash} not found"},
            status_code=status.HTTP_404_NOT_FOUND
        )
    return ArticleResponse(**results)


@search_articles_router.post(
    "/make-hashed-link",
    status_code=status.HTTP_200_OK,
    response_model=LinkResponse,
    operation_id="make_hashed_link",
)
async def make_hashed_link(link: Link):
    """
    Creates a hashed version of the provided link using the specified hashing function.
    This function expects a Link object as input and returns a response containing
    the original link and its hashed value.

    :param link: The link object containing the URL to be hashed
    :type link: Link
    :return: A response object containing the original link and its hashed equivalent
    :rtype: LinkResponse
    """
    return LinkResponse(**{
        "link": link.link,
        "link_hash": make_hashed_url(str(link.link))
    })
