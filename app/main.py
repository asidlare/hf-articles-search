import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from app.config import config
from app.services.database import databasemanager
from app.routers.search_articles import search_articles_router

def init_app(init_db=True):
    lifespan = None

    if init_db:
        databasemanager.init(config.DB_CONFIG)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            if databasemanager._engine is not None:
                await databasemanager.close()

    serv = FastAPI(title="FastAPI test server", lifespan=lifespan)
    serv.include_router(search_articles_router)

    return serv


my_app = init_app()


# Initialize FastAPI-MCP
included_operations = [
    "search_articles_by_tag_names",
    "get_article_by_link_hash",
    "make_hashed_link"
]
mcp_serv = FastApiMCP(
    my_app,
    http_client=httpx.AsyncClient(timeout=60),
    name="API for searching articles from science category from Huffington Post",
    description="""
        MCP server for searching articles from science category from Huffington Post.
        Exposes the following tools:
        - search_articles_by_tag_names
        - get_article_by_link_hash
        - make_hashed_link
    """,
    include_operations=included_operations,
    describe_full_response_schema=True,
    describe_all_responses=True,
)

# Mount both the MCP server and SSE server
mcp_serv.mount()  # Mount MCP to main app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(my_app, host="0.0.0.0", port=8000, timeout_keep_alive=60)
