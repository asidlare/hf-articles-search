from app.api.data_upload import upload_articles, get_articles_by_link_hash


async def test_upload_articles(
    science_category_file,
    science_category_llm_summaries_file,
    link_1_hash,
    link_2_hash,
    generate_embedding,
):
    await upload_articles(
        category_file=science_category_file,
        summary_file=science_category_llm_summaries_file,
    )

    article1 = await get_articles_by_link_hash(link_hash=link_1_hash)
    # article itself
    assert article1 is not None
    assert article1.headline == "First Article Title"
    assert article1.date.isoformat() == "2024-01-02"
    # embedding
    assert article1.embedding is not None
    assert article1.embedding.summarization == "It is a summary of the first article."
    assert article1.embedding.embedding is not None
    # tags
    assert len(article1.tags or []) == 3
    tags = [tag.tag for tag in article1.tags]
    assert "tag1" in tags
    assert "new-tag" in tags
    assert "tag2" in tags
    # key insights
    assert len(article1.key_insights or []) == 2
    key_insights = [key_insight.key_insight for key_insight in article1.key_insights]
    assert "First insight." in key_insights
    assert "Second insight." in key_insights

    article2 = await get_articles_by_link_hash(link_hash=link_2_hash)
    # article itself
    assert article2 is not None
    assert article2.headline == "Second Article Title"
    assert article2.date.isoformat() == "2024-01-03"
    # embedding
    assert article2.embedding is not None
    assert article2.embedding.summarization == "It is a summary of the second article."
    assert article2.embedding.embedding is not None
    # tags
    assert len(article2.tags or []) == 3
    tags = [tag.tag for tag in article2.tags]
    assert "tag1" in tags
    assert "next-tag" in tags
    assert "tag3" in tags
    # key insights
    assert len(article2.key_insights or []) == 2
    key_insights = [key_insight.key_insight for key_insight in article2.key_insights]
    assert "Third insight." in key_insights
    assert "Fourth insight." in key_insights
