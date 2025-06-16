import os

class Config:
    DB_CONFIG = os.getenv(
        "DB_CONFIG",
        "postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_NAME}".format(
            POSTGRES_USER=os.getenv("POSTGRES_USER"),
            POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD"),
            POSTGRES_HOST=os.getenv("POSTGRES_HOST"),
            POSTGRES_NAME=os.getenv("POSTGRES_NAME"),
        ),
    )


config = Config
