import asyncio
import typer
from app.transformations.extractor import extract_science_category
from app.transformations.scrapper import fetch_urls


app = typer.Typer()


@app.command()
def extract_science_category_from_json():
    typer.echo("Starting...")
    extract_science_category()
    typer.echo("Finished!")


@app.command()
def fetch_urls_from_json_file(input_file: str, output_file: str):
    typer.echo(f"Starting fetching urls from {input_file} and saving to {output_file}...")
    asyncio.run(fetch_urls(input_file, output_file))
    typer.echo("Finished!")


if __name__ == "__main__":
    app()
