import asyncio
import typer
from app.api.data_upload import upload_articles
from app.main import init_app

app = typer.Typer()


@app.command()
def upload_data_from_json_file():
    typer.echo("Starting...")
    fastapi_app = init_app()
    try:
        asyncio.run(upload_articles())
    finally:
        pass

    typer.echo("Finished!")


if __name__ == "__main__":
    app()
