import openai
import os
from datetime import datetime
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column


openai.api_key = os.getenv("OPENAI_API_KEY")


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class AutoEmbeddingMixin:
    embedding: Mapped[list[float]] = mapped_column(VECTOR(1536))

    # Define which columns should be used for embedding
    @property
    def embedding_column(self) -> str:
        raise NotImplementedError(
            "Define embedding_columns in your model"
        )

    def get_text_for_embedding(self) -> str:
        return getattr(self, self.embedding_column)


def generate_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI API"""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
