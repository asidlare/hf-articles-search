from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import config
from app.services.database import databasemanager
# from app.routers_v1 import router_v1


def init_app(init_db=True):
    lifespan = None

    if init_db:
        databasemanager.init(config.DB_CONFIG)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            if databasemanager._engine is not None:
                await databasemanager.close()

    server = FastAPI(title="FastAPI test server", lifespan=lifespan)
    # server.include_router(router_v1, prefix="/v1", tags=["v1"])

    return server


my_app = init_app()
