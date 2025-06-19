from datetime import date
from sqlalchemy.dialects.postgresql import DATE
from sqlalchemy import BIGINT, VARCHAR, CHAR, ForeignKey, UniqueConstraint, event
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from typing import Optional

from app.models.utils import AutoEmbeddingMixin, TimestampMixin, generate_embedding
from app.services.database import Base


class ArticleTag(Base, TimestampMixin):
    __tablename__ = 'article_tags'

    # Composite primary key consisting of foreign keys
    article_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey('articles.article_id'), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey('tags.tag_id'), primary_key=True
    )

    # Relationships back to the Student and Course models
    # These create the direct links from the association object
    article: Mapped["Article"] = relationship(back_populates="tags")
    tag: Mapped["Tag"] = relationship(back_populates="tags")


class Article(Base, TimestampMixin):
    __tablename__ = "articles"

    article_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    link_hash: Mapped[str] = mapped_column(CHAR(16), unique=True, nullable=False)
    link: Mapped[str] = mapped_column(VARCHAR(255), unique=True, nullable=False)
    headline: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    date: Mapped[date] = mapped_column(DATE, nullable=False)

    # One-to-one relationship with Embedding
    embedding: Mapped[Optional["Embedding"]] = relationship(back_populates="article", uselist=False)

    # Many-to-many relationship with Tag
    # Relationship to the association object (one-to-many from Article to ArticleTage)
    article_tags: Mapped[list["ArticleTag"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )
    tags: Mapped[list["Tag"]] = relationship(
        secondary="article_tags",  # Refer to the table name of the association object
        primaryjoin="Article.article_id == ArticleTag.article_id",
        secondaryjoin="Tag.tag_id == ArticleTag.tag_id",
        viewonly=True  # Prevents write operations through this relationship, use 'article_tags' instead
    )

    # One-to-many relationship
    key_insights: Mapped[list["KeyInsight"]] = relationship(back_populates="article", cascade="all, delete-orphan")


class Embedding(Base, TimestampMixin, AutoEmbeddingMixin):
    __tablename__ = "embeddings"

    embedding_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    summarization: Mapped[str] = mapped_column(VARCHAR(1000), nullable=False)

    article_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey(
            "articles.article_id",
            ondelete="CASCADE"
        ),
        index=True,
        nullable=False,
    )
    article: Mapped[Article] = relationship(back_populates="embedding")

    @property
    def embedding_column(self):
        return 'summarization'


class Tag(Base, TimestampMixin):
    __tablename__ = "tags"

    tag_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    tag: Mapped[str] = mapped_column(VARCHAR(255), unique=True, nullable=False, index=True)

    # Relationship to the association object (one-to-many from Tag to ArticleTag)
    # This represents all article tags in a tag
    article_tags: Mapped[list["ArticleTag"]] = relationship(
        back_populates="tag", cascade="all, delete-orphan"
    )

    # Convenience relationship to access Articles directly through enrollments
    articles: Mapped[list["Article"]] = relationship(
        secondary="article_tags",  # Refer to the table name of the association object
        primaryjoin="Tag.tag_id == ArticleTag.tag_id",
        secondaryjoin="Article.article_id == ArticleTag.article_id",
        viewonly=True  # Prevents write operations through this relationship, use 'article_tags' instead
    )


class KeyInsight(Base, TimestampMixin):
    __tablename__ = "key_insights"
    __table_args__ = (
        UniqueConstraint('key_insight', 'article_id'),
    )

    key_insight_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    key_insight: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    article_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey(
            "articles.article_id",
            ondelete="CASCADE"
        ),
        index=True,
        nullable=False,
    )

    # Many-to-one relationship
    article: Mapped["Article"] = relationship(back_populates="key_insights")


# Event listener for the mixin
@event.listens_for(Session, 'before_flush')
def generate_embeddings_before_flush(session, flush_context, instances):
    for obj in session.new | session.dirty:
        if isinstance(obj, AutoEmbeddingMixin):
            # Check if the embedding source column was modified
            modified = False
            if obj in session.new:
                modified = True
            elif obj in session.dirty:
                if session.is_modified(obj, [obj.embedding_column]):
                    modified = True

            if modified:
                text = obj.get_text_for_embedding()
                obj.embedding = generate_embedding(text)
