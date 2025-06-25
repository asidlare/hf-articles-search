from datetime import date
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


class ArticleResponse(BaseModel):
    """
    Representation of an article's response containing its metadata, summary, and associated insights.

    This class is designed to encapsulate the response corresponding to an article,
    providing information like its URL, headline, published date, a summarization,
    key insights derived from the article, and related tag names. Each attribute
    offers specific metadata to comprehensively define the article and its context.
    """
    link: HttpUrl = Field(description="URL to the article")
    link_hash: str = Field(description="Hash of the article URL link")
    headline: str = Field(description="Headline of the article")
    published_date: date = Field(description="Published date of the article")
    summarization: str = Field(description="Summarization of the article")
    key_insights: list[str] = Field(description="Key insights of the article")
    tag_names: list[str] = Field(description="Tag names defined for the article")


class ExtendedArticleResponse(BaseModel):
    """
    Represents an extended response for an article, including information such as position
    ID, URL link, headline, published date, summarization, and analysis details like key
    insights and tag-related metadata.

    The purpose of this class is to provide a detailed representation of an article in
    a structured format that includes both general article information and its relationships
    to semantic tags and embedding-based calculations.
    """
    article_position_id: int = Field(description="A position id for the article")
    link: HttpUrl = Field(description="URL to the article")
    link_hash: str = Field(description="Hash of the article URL link")
    headline: str = Field(description="Headline of the article")
    published_date: date = Field(description="Published date of the article")
    summarization: str = Field(description="Summarization of the article")
    key_insights: list[str] = Field(description="Key insights of the article")
    tag_names: Optional[list[str]] = Field(default=None, description="Tag names defined for the article")
    embedding_position_id: Optional[int] = Field(
        default=None,
        description="A position id based on semantic search based on tag names"
    )
    distance: float = Field(description="Cosine distance between the article and the tags embedding")


class TagSearchResponse(BaseModel):
    """
    Represents the response containing a list of articles found for specified tag names.

    This class is a model that encapsulates the response structure when querying for
    articles associated with specific tag names. It includes details about the
    articles found during the search operation.
    """
    articles: list[ExtendedArticleResponse] = Field(description="A list of articles found for specified tag names")


class Link(BaseModel):
    """
    Represents a hyperlink pointing to an article.

    This class is a model for storing a valid HTTP URL linking to an article.
    It ensures that the provided URL adheres to the expected format of an HTTP
    or HTTPS URL. The model is intended to provide structured validation and
    simplification of working with article links, ensuring correctness and
    consistency.
    """
    link: HttpUrl = Field(description="URL to the article")


class LinkResponse(Link):
    """
    Represents a response model for a Link.

    This class extends the Link class and adds an additional attribute
    `link_hash` which represents the hash of the article URL link. It is
    specifically tailored to include more details about a link within a
    certain context, such as when dealing with URL processing or API
    responses.
    """
    link_hash: str = Field(description="Hash of the article URL link")
