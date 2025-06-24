from app.api.data_upload import upload_articles


async def test_search_articles_by_tag_names(
    client,
    science_category_extended_file,
    science_category_llm_summaries_extended_file,
    science_category_extended_data,
    generate_embedding,
):
    await upload_articles(
        category_file=science_category_extended_file,
        summary_file=science_category_llm_summaries_extended_file,
    )

    # searching animal tags
    result = client.get("/search-articles-by-tag-names?tag_names=animal,zoo")
    assert result.status_code == 200
    animal_hashes = [row["link_hash"] for row in science_category_extended_data[:12]]
    for row in result.json()["articles"]:
        assert row["link_hash"] in animal_hashes

    first_row_keys = list(result.json()["articles"][0].keys())
    assert "article_position_id" in first_row_keys
    assert "link" in first_row_keys
    assert "link_hash" in first_row_keys
    assert "headline" in first_row_keys
    assert "published_date" in first_row_keys
    assert "summarization" in first_row_keys
    assert "tag_names" in first_row_keys
    assert "key_insights" in first_row_keys
    assert "embedding_position_id" in first_row_keys
    assert "distance" in first_row_keys

    # strange tag names, still results from embeddings search
    result = client.get("/search-articles-by-tag-names?tag_names=hector,achilles")
    assert result.status_code == 200
    assert all(row["embedding_position_id"] is not None for row in result.json()["articles"])

    # no tag_names
    result = client.get("/search-articles-by-tag-names")
    assert result.status_code == 422

    # empty tag_names
    result = client.get("/search-articles-by-tag-names?tag_names=")
    assert result.status_code == 400


async def test_get_article_by_link_hash(
    client,
    science_category_extended_file,
    science_category_llm_summaries_extended_file,
    science_category_extended_data,
    llm_summaries_extended_data,
    link_1_hash,
    generate_embedding,
):
    await upload_articles(
        category_file=science_category_extended_file,
        summary_file=science_category_llm_summaries_extended_file,
    )

    article_category = science_category_extended_data[0]
    article_llm_summary = llm_summaries_extended_data[0]

    # fetching first article
    result = client.get(f"/get-article-by-link-hash?link_hash={article_category['link_hash']}")
    assert result.status_code == 200

    data = result.json()
    assert data["link"] == article_category["link"]
    assert data["link_hash"] == article_category["link_hash"]
    assert data["headline"] == article_category["headline"]
    assert data["published_date"] == article_category["date"]

    assert data["summarization"] == article_llm_summary["summarization"]
    assert len(data["tag_names"]) == len(article_llm_summary["tags"])
    for tag_name in article_llm_summary["tags"]:
        assert tag_name in data["tag_names"]
    assert len(data["key_insights"]) == len(article_llm_summary["key_insights"])
    for key_insight in article_llm_summary["key_insights"]:
        assert key_insight in data["key_insights"]

    # non-existing link_hash
    result = client.get(f"/get-article-by-link-hash?link_hash={link_1_hash}")
    assert result.status_code == 404

    # no link_hash
    result = client.get("/get-article-by-link-hash")
    assert result.status_code == 422

    # empty link hash
    result = client.get("/get-article-by-link-hash?link_hash=")
    assert result.status_code == 400
