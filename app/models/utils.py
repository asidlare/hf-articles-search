import openai
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy import Index, func
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


def generate_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI API"""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
