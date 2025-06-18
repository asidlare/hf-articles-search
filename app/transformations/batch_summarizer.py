import openai
import json
import pandas as pd
import time
import os
from itertools import islice
from tqdm import tqdm
from typing import List, Optional, Literal
from tqdm.std import tqdm
from app.schemas.batch_response_output import SummaryResponseOutput

# path to data folder
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
# llm settings
MODEL_NAME = "gpt-4.1-mini"
openai.api_key = os.environ['OPENAI_API_KEY']


SYSTEM_MESSAGE = f"""
You are an advanced article analysis system designed to process and structure content from articles.
Your task is to analyze the provided article text and generate a structured response that strictly 
adheres to the following format and guidelines: {SummaryResponseOutput.model_json_schema()}

STRUCTURAL REQUIREMENTS:
1. Output must be valid JSON
2. All strings must use double quotes, not single quotes
3. No trailing commas in arrays or objects
4. No comments or additional text outside the JSON structure
5. Use UTF-8 encoded characters only
6. Escape special characters properly (\n, \", etc.)

CONTENT GUIDELINES:

GENERAL CONTENT CLEANUP BEFORE ANALYSIS:
- Remove all navigation menus, social media buttons, and sidebar content
- Remove all advertisements and promotional content
- Remove all footer elements and legal notices
- Remove special characters and normalize quotes/apostrophes
- Keep only the actual article content that would appear in the main reading area
- Do not include "Read more" or related article links
- Remove any newsletter signup forms or call-to-action elements

Summarization:
- Write 2-10 paragraphs (100-500 words total - depending on the length of the article)
- Use clear, professional language
- Present information chronologically or by importance
- Focus on main points and conclusions
- Avoid repetition and filler words
- Use complete sentences

Tags:
- Include 3-10 relevant tags
- Use lowercase unless proper nouns
- No special characters or spaces (use hyphens if needed)
- Keep tags concise (1-3 words)
- Order from most to least relevant
- Include domain/topic-specific tags

Key Insights:
- List 3-10 significant takeaways
- Start each insight with a capital letter
- End each insight with a period
- Keep each insight to 1-2 sentences
- Focus on actionable or noteworthy information
- Avoid redundancy with other insights
- Present in order of importance

QUALITY REQUIREMENTS:
1. Grammar and spelling must be impeccable
2. Maintain consistent style and tone
3. Use Oxford comma in lists
4. Write numbers under 10 as words
5. Use active voice when possible
6. Avoid colloquialisms and jargon
7. Maintain objectivity

ERROR PREVENTION:
1. Verify JSON structure before output
2. Check for balanced quotes and brackets
3. Validate character count limits
4. Ensure no null or undefined values
5. Verify minimum and maximum list lengths
6. Check for accidental HTML/markdown formatting
7. Validate proper string escaping

STRICTLY FORBIDDEN:
1. Adding extra fields not in the schema
2. Including metadata or notes outside JSON
3. Using formatting like bold or italics
4. Adding line numbers or indices
5. Including source URLs or references
6. Adding timestamps or version information
7. Including personal opinions or commentary

If you understand these instructions, process the provided article text and generate a response
following these exact specifications.
If any part of the article is unclear or ambiguous, focus on the most definitive information available
while maintaining the required structure and quality standards.
"""


def prepare_batch_file(chunk: list[str], chunk_output_file: str, pbar: tqdm) -> None:
    """
    Processes a batch of input data, parses each entry, and prepares API requests for summarization
    in a specified output file. Ensures malformed or insufficient content entries are skipped, and
    generates uniquely identifiable request payloads for valid entries. Progress updates are displayed
    via the provided progress bar.

    :param chunk: A list of JSON-formatted strings, each representing a data entry to process.
    :param chunk_output_file: The file path where processed API request data will be written.
    :param pbar: A progress bar instance used to track and display the processing progress.
    :return: None
    """
    requests_prepared = 0

    # Clear previous chunk batch input file
    if os.path.exists(chunk_output_file):
        os.remove(chunk_output_file)

    print(f"Processing chunk_output_file: {chunk_output_file}, number of lines: {len(chunk)}")
    with open(chunk_output_file, "w") as outfile:
        for line in chunk:
            try:
                data = json.loads(line.strip())
                link = data.get("link")
                raw_content = data.get("content")
                link_hash = data.get("link_hash")

                if not all([link, raw_content, link_hash]):
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

                # Only process if there's substantial text content
                if len(raw_content) < 100:  # Adjust min length as needed
                    print(f"Skipping short content from {link}")
                    continue

                # custom_id created using link hash and counter to ensure uniqueness
                custom_id = f"summary_for_{link_hash}_{str(requests_prepared)}"

                request_data = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [
                            {
                                "role": "system",
                                "content": SYSTEM_MESSAGE
                            },
                            {
                                "role": "user",
                                "content": f"Summarize the following web article content:\n\n{raw_content}"
                            }
                        ],
                        "temperature": 0.3,  # Low temperature for factual, consistent output
                        "response_format": {"type": "json_object"}  # Crucial for enforcing JSON output
                    }
                }
                outfile.write(json.dumps(request_data) + "\n")
                requests_prepared += 1
            except json.JSONDecodeError:
                print(f"Error decoding JSON from scraped file: {line.strip()}")
            except Exception as e:
                print(f"Unexpected error preparing request: {e} - Line: {line.strip()}")

    pbar.update(len(chunk))
    print(f"Prepared {requests_prepared} requests in '{chunk_output_file}'.")

    if requests_prepared == 0:
        return None


def prepare_batch_requests(scraped_file: str, batch_output_file: str) -> Optional[str]:
    """
    Prepares and processes a file into smaller batch files for further operations. The function
    reads a specified file, divides its content into chunks, and generates batch files which
    can be used for OpenAI requests or other processing tasks. The batch files are written to
    a specific output directory.

    :param scraped_file: The name of the input file containing scraped data to be divided into chunks.
    :param batch_output_file: The target output file name to store generated batch files.
                              The actual batch files will include chunk numbers in their names.
    :return: The function returns None if the input file is not found; otherwise, it returns
             no value explicitly after generating the batches.
    :rtype: Optional[str]
    """
    scraped_file = os.path.join(DATA_PATH, scraped_file)
    batch_output_file = os.path.join(DATA_PATH, 'batch', batch_output_file)

    if not os.path.exists(scraped_file):
        print(f"Error: Scraped data file '{scraped_file}' not found.")
        return None

    print(f"Preparing batch requests from '{scraped_file}' for OpenAI...")
    chunk_number = 0
    total_lines = sum(1 for _ in open(scraped_file, 'r'))

    with open(scraped_file, 'r') as infile:
        with tqdm(total=total_lines, desc="Processing file") as pbar:
            while True:
                chunk = list(islice(infile, 50))
                if not chunk:
                    break

                chunk_output_file = batch_output_file.replace(".jsonl", f"_{chunk_number}.jsonl")
                prepare_batch_file(chunk, chunk_output_file, pbar)
                chunk_number += 1


def upload_batch_input_file(batch_output_file: str) -> Optional[str]:
    """
    Uploads the input batch file to OpenAI's file API and returns the file ID if
    successful.

    The function takes the name of a batch output file, constructs the full
    file path, and uploads the file with the purpose "batch" via the OpenAI API.
    If the upload is successful, the function returns the unique file ID of the
    uploaded file. In the event of an API error, this function logs the error
    message and returns None.

    :param batch_output_file: The name of the batch output file to be uploaded.
    :type batch_output_file: str
    :return: The unique ID of the uploaded file if successful, or None if an
             error occurs during the upload.
    :rtype: Optional[str]
    """
    # Upload the input file
    batch_output_file = os.path.join(DATA_PATH, 'batch', batch_output_file)
    print("Uploading the batch input file to OpenAI...")
    try:
        batch_input_file_obj = openai.files.create(
            file=open(batch_output_file, "rb"),
            purpose="batch"
        )
        print(f"File uploaded. File ID: {batch_input_file_obj.id}")
        return batch_input_file_obj.id
    except openai.APIError as e:
        print(f"Error uploading batch input file: {e}")
        return None


def run_batch_job(input_file_id: str, chunk_number: int) -> Optional[str]:
    """
    Creates and monitors a batch job using OpenAI's API.

    The function handles the creation and polling of a batch job, which processes chunks
    of data associated with the provided input file. It waits asynchronously for the
    batch job to complete and retrieves its output file ID if successful. In the event
    that the batch job fails, is cancelled, or encounters errors, the errors are logged,
    and the function returns None.

    :param input_file_id: The ID of the input file to be processed in the batch job.
    :type input_file_id: str
    :param chunk_number: The chunk number being currently processed within the batch job.
    :type chunk_number: int
    :return: The output file ID if the batch job completes successfully, otherwise None.
    :rtype: Optional[str]
    """
    print("Creating the batch job...")
    try:
        batch_job = openai.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"  # Or "1h" depending on your needs and availability
        )
        print(f"Batch job created. Job ID: {batch_job.id}")
        print(f"Batch job status: {batch_job.status}")
    except openai.APIError as e:
        print(f"Error creating batch job: {e}")
        return None

    print("Polling for batch job completion (this may take some time)...")
    check_number = 1
    while True:
        try:
            current_batch_job = openai.batches.retrieve(batch_job.id)
            print(f"Current batch job status: {current_batch_job.status}, chunk: {chunk_number}, check: {check_number}")
            check_number += 1

            if current_batch_job.status in ["completed", "failed", "cancelled"]:
                if current_batch_job.status == "completed":
                    print("Batch job completed successfully!")
                    return current_batch_job.output_file_id
                else:
                    print(f"Batch job finished with status: {current_batch_job.status}")
                    if current_batch_job.errors:
                        print("Errors encountered in batch:")
                        for error in current_batch_job.errors.data:
                            print(f"  Code: {error.code}, Message: {error.message}")
                    return None

            time.sleep(30)  # Wait 30 seconds before polling again
        except openai.APIError as e:
            print(f"Error retrieving batch job status: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during polling: {e}")
            return None


def download_batch_results(output_file_id: str, chunk_number: int) -> None:
    """
    Downloads and processes the batch results from a specified output file ID and saves it
    to a specified location. Handles errors that may occur during the download and file
    writing process.

    :param output_file_id: The unique identifier for the batch result file to download. This
        is used to fetch batch file content from OpenAI's API.
    :type output_file_id: str
    :param chunk_number: The chunk number used to create the final output file name for
        saving the batch result locally.
    :type chunk_number: int
    :return: None
    """
    print("\nDownloading and processing batch results...")
    try:
        result_content = openai.files.content(output_file_id).read()
        final_output_jsonl = os.path.join(DATA_PATH, 'llm', f"summarized_articles_{chunk_number}.jsonl")
        with open(final_output_jsonl, "wb") as f:
            f.write(result_content)
        print(f"Raw results downloaded to: {final_output_jsonl}")

    except openai.APIError as e:
        print(f"Error downloading batch output file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during result processing: {e}")


def parse_structured_summaries(output_summary_dir: str, output_summary_file: str):
    """
    Parses and processes structured summary files located within a given directory.

    This function reads JSON lines from summary files, extracts structured content including
    `tags`, `key_insights`, and `summarization`, validates them with a specified schema, and
    aggregates the results into a DataFrame. Also handles error cases, such as JSON decoding
    issues, missing or malformed data, and API errors, while logging relevant information.
    The final processed summaries are saved as a JSON file in the specified output location.

    :param output_summary_dir: Directory containing summary files to process.
    :type output_summary_dir: str
    :param output_summary_file: Name of the output file to save processed summaries in JSON format.
    :type output_summary_file: str
    :return: None
    """
    print("\n--- Parsing Structured Summaries ---")
    successful_summaries = 0
    output_data = []
    for summary_file in os.listdir(output_summary_dir):
        summary_file_path = os.path.join(output_summary_dir, summary_file)
        with open(summary_file_path, "r") as f:
            for line in f:
                output = {"tags": [], "key_insights": []}
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "N/A")
                    output["link_hash"] = custom_id.split("_")[2]
                    output["order_id"] = custom_id.split("_")[-1]
                    output["custom_id"] = custom_id

                    if result.get("response") and result["response"].get("body"):
                        choices = result["response"]["body"].get("choices")
                        if choices and len(choices) > 0:
                            raw_content = choices[0]["message"]["content"].strip()

                            try:
                                parsed_summary = SummaryResponseOutput.model_validate_json(raw_content)
                                output["summarization"] = parsed_summary.summarization
                                for tag in parsed_summary.tags:
                                    output["tags"].append(tag)
                                for key_insight in parsed_summary.key_insights:
                                    output["key_insights"].append(key_insight)
                                output_data.append(output)

                                successful_summaries += 1
                            except Exception as parse_error:
                                print(
                                    f"Request ID: {custom_id}\n  Error parsing JSON from LLM (output might not conform to schema): {parse_error}\nRaw Content:\n{raw_content}\n---")
                        else:
                            print(
                                f"Request ID: {custom_id}\n  Error: No choices found in response for this request. Raw response: {result.get('response', {}).get('body')}\n---")
                    elif result.get("error"):
                        error_detail = result["error"]
                        print(
                            f"Request ID: {custom_id}\n  API Error for this request: {error_detail.get('message', 'Unknown error')}\n  Code: {error_detail.get('code', 'N/A')}\n---")
                    else:
                        print(f"Request ID: {custom_id}\n  Unexpected result structure: {result}\n---")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line from batch output: {line.strip()}")
    output_df = pd.DataFrame(output_data)
    output_summary_file = os.path.join(DATA_PATH, output_summary_file)
    output_df.to_json(output_summary_file, orient="records", lines=True, index=False)
    print(f"\nSuccessfully processed {successful_summaries} structured summaries.")
