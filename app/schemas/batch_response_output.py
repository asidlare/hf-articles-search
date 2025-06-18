from pydantic import BaseModel, Field


class SummaryResponseOutput(BaseModel):
    summarization: str = Field(
        ...,
        description="A concise summary of the article which is embedded in the provided website content."
    )
    tags: list[str] = Field(
        ...,
        description="A list of 3-10 tags which can be used to categorize the article."
    )
    key_insights: list[str] = Field(
        ...,
        description="A list of 3-10 key insights extracted from the article text."
    )
