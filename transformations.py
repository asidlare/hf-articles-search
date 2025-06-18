import asyncio
import typer
import os
from app.transformations.batch_summarizer import (
    prepare_batch_requests,
    upload_batch_input_file,
    run_batch_job,
    download_batch_results,
    parse_structured_summaries
)
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
    try:
        asyncio.run(fetch_urls(input_file, output_file))
    except KeyboardInterrupt:
        print("Scraping interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    typer.echo("Finished!")


@app.command()
def prepare_batch_requests_for_openai(input_file: str, output_file: str):
    typer.echo(f"Starting preparing batch requests from {input_file} and saving to {output_file}...")
    try:
        prepare_batch_requests(input_file, output_file)
    except KeyboardInterrupt:
        print("Scraping interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    typer.echo("Finished!")


@app.command()
def process_batch_requests(
        batch_file_dir: str,
        output_summary_dir: str,
        output_summary_file: str = "science_category_llm_summaries.jsonl"
):
    typer.echo("Starting Batch Summarization Workflow...")

    if not (os.path.exists(output_summary_dir) and os.path.isdir(output_summary_dir)):
        raise ValueError(f"Output directory {output_summary_dir} does not exist or is not a directory.")
    if not (os.path.exists(batch_file_dir) and os.path.isdir(batch_file_dir)):
        raise ValueError(f"Output directory {batch_file_dir} does not exist or is not a directory.")

    for batch_input_file in os.listdir(batch_file_dir):
        chunk_number = batch_input_file.split(".")[0].split("_")[-1]
        uploaded_file_id = upload_batch_input_file(batch_input_file)

        if uploaded_file_id:
            # Step 2: Run the batch job
            output_file_id = run_batch_job(uploaded_file_id, chunk_number)

            if output_file_id:
                # Step 3: Process the batch results
                download_batch_results(output_file_id, chunk_number)
            else:
                print("Batch job did not produce an output file. Check logs for errors.")
        else:
            print("No batch input file was uploaded. Exiting.")

    parse_structured_summaries(output_summary_dir, output_summary_file)
    typer.echo("Batch Summarization Workflow Complete...")


if __name__ == "__main__":
    app()
