import asyncio
import aiohttp
import pandas as pd
import os
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# path to data folder
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
# Limit the number of concurrent HTTP requests
MAX_CONCURRENT_REQUESTS = 50
# Use more threads than CPU cores for I/O-bound tasks like parsing
MAX_PARSING_WORKERS = os.cpu_count() * 2 if os.cpu_count() else 4


# --- Async HTTP Fetching ---
async def fetch_html(session: aiohttp.ClientSession, url: str) -> str:
    """
    Fetches the HTML content of a given URL asynchronously using an aiohttp session.

    This function makes a GET request to the specified URL and retrieves the HTML
    content. If the request fails due to an HTTP error (such as 4xx or 5xx) or
    other client-side errors, it returns an empty string to ensure the calling
    code can handle errors gracefully.

    :param session: An active aiohttp.ClientSession instance used to make the HTTP request.
    :type session: aiohttp.ClientSession
    :param url: The URL from which the HTML content will be fetched.
    :type url: str
    :return: The HTML content of the page retrieved from the URL, or an empty
        string if the request fails.
    :rtype: str
    """
    try:
        # added a timeout
        async with session.get(url, timeout=15) as response:
            # raise an exception for Http errors (4xx or 5xx)
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        print(f"Error fetching {url}: {e}")
        # return empty string on error to allow parsing to handle gracefully
        return ""


# --- Blocking Parsing Function ---
def parse_html_blocking(html_content: str, url: str) -> dict[str, str]:
    """
    Parses an HTML document to extract its text content in a blocking manner.

    This function takes an HTML content string alongside its URL and processes
    the HTML to extract readable text. It returns a dictionary containing the
    source URL and the extracted content or an error message in case of parsing
    fails. The parsing is performed synchronously and uses BeautifulSoup for
    HTML parsing.

    :param html_content: The HTML content to be parsed.
    :type html_content: str
    :param url: The URL corresponding to the HTML content.
    :type url: str
    :return: A dictionary containing the source URL and extracted text content
        or an error message in case of failure.
    :rtype: dict[str, str]
    """
    if not html_content:
        return {"url": url, "error": "No content to parse"}

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract text content
        text_content = soup.get_text(separator=" ", strip=True)

        return {
            "link": url,
            "content": text_content
        }
    except Exception as e:
        print(f"Error parsing HTML for {url}: {e}")
        return {"url": url, "error": f"Parsing failed: {e}"}


# --- Main Async Scraper Logic ---
async def scrape_url(session: aiohttp.ClientSession, url: str, executor: ThreadPoolExecutor) -> dict:
    """
    Fetches and parses the HTML content of a given URL asynchronously using aiohttp and
    a ThreadPoolExecutor. The function involves an asynchronous HTTP request to retrieve
    HTML content and uses a blocking BeautifulSoup parsing operation executed within an executor.

    This function is useful for scenarios where asynchronous and blocking operations need to
    be combined, such as parsing large HTML documents in a non-blocking application.

    :param session: An active aiohttp ClientSession to perform HTTP requests.
    :type session: aiohttp.ClientSession
    :param url: The URL of the webpage to fetch and parse.
    :type url: str
    :param executor: An instance of ThreadPoolExecutor to execute the blocking HTML parsing.
    :type executor: ThreadPoolExecutor
    :return: A dictionary containing parsed data or an error message if HTML fetching fails.
    :rtype: dict
    """
    html_content = await fetch_html(session, url)
    if not html_content:
        return {"url": url, "status": "Failed to fetch"}

    # Use run_in_executor for the blocking BeautifulSoup parsing
    loop = asyncio.get_running_loop()
    parsed_data = await loop.run_in_executor(
        executor,
        parse_html_blocking,
        html_content,
        url # Pass the URL to the parsing function for context
    )
    return parsed_data


def read_urls_from_file(filename: str)-> list[str]:
    """
    Reads a list of URLs from a JSON line file specified by the given filename.
    Verifies that the file has a `.jsonl` extension and exists in the predefined
    data path. If the file's format or existence does not comply with the
    requirements, appropriate exceptions are raised.

    :param filename: The name of the JSON line file containing the URLs.
    :type filename: str

    :raises ValueError: If the filename does not end with `.jsonl`.
    :raises FileNotFoundError: If the file with the specified name does not
        exist in the expected location.

    :return: A list of URLs extracted from the JSON line file.
    :rtype: list[str]
    """
    if not filename.endswith(".jsonl"):
        raise ValueError("Filename must end with .jsonl")

    FILE_PATH = os.path.join(DATA_PATH, filename)
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File {filename} not found in {DATA_PATH}.")
    return pd.read_json(FILE_PATH, lines=True)["link"].tolist()


def save_to_jsonl(data: dict, filename):
    """
    Saves a dictionary as a single line in a JSONL file. JSONL, or JSON Lines, is a
    format for storing structured data in which each line is a separate and
    independent JSON object.

    :param data: The dictionary to be saved as a JSON object.
    :type data: dict
    :param filename: The name of the JSONL file where the data will be appended. This
        must end with the ".jsonl" file extension.
    :type filename: str
    :return: None
    """
    if not filename.endswith(".jsonl"):
        raise ValueError("Filename must end with .jsonl")

    FILE_PATH = os.path.join(DATA_PATH, filename)
    with open(FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"{data}\n")


async def fetch_urls(input_file: str, output_file: str = "science_links_content.jsonl") -> None:
    """
    Fetches URLs from the input file, performs asynchronous scraping, and writes the
    results to the specified output file in JSON Lines format. The scraping process
    is highly concurrent while limiting the number of simultaneous HTTP requests
    through the use of a semaphore. The parsed content of pages is processed using
    a thread-safe thread pool executor to handle any blocking operations effectively.

    The function reads a list of URLs from the input file, fetches and parses the
    content from each URL concurrently, and saves the results incrementally in the
    output file to reduce memory usage and enable progress tracking during large-scale
    scraping operations.

    :param input_file: Path to the input file containing URLs, one per line.
    :type input_file: str
    :param output_file: Path to save the scraped JSONL content. Defaults to
        "science_links_content.jsonl".
    :type output_file: str, optional
    :return: This function does not return a value. All results are directly written
        to the output file.
    :rtype: None
    """
    start_time = time.time()
    urls = read_urls_from_file(input_file)
    results = 0

    # Initialize a ThreadPoolExecutor for blocking parsing tasks
    # Using ThreadPoolExecutor as parsing is often I/O-bound (string processing) and
    # the GIL is released by BeautifulSoup's C extensions.
    print(f"Using ThreadPoolExecutor with {MAX_PARSING_WORKERS} workers for parsing.")
    with ThreadPoolExecutor(max_workers=MAX_PARSING_WORKERS) as parser_executor:
        async with aiohttp.ClientSession() as session:
            # Create a list of tasks for fetching and parsing
            tasks = [scrape_url(session, url, parser_executor) for url in urls]

            # Use asyncio.BoundedSemaphore to limit concurrent HTTP requests
            semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

            # Define an async wrapper to acquire/release the semaphore
            async def limited_task(task):
                async with semaphore:
                    return await task

            # Gather results concurrently with the semaphore limitation
            print(f"Starting scraping {len(urls)} URLs with {MAX_CONCURRENT_REQUESTS} concurrent requests.")
            for i, future in enumerate(asyncio.as_completed([limited_task(t) for t in tasks])):
                result = await future
                results += 1
                if (i + 1) % 50 == 0: # Print progress every 50 URLs
                    print(f"Processed {i + 1}/{len(urls)} URLs.")
                # You might want to save results incrementally here if it's a very large scrape
                save_to_jsonl(result, filename=output_file)

    end_time = time.time()
    print(f"\n--- Scrape Complete ---")
    print(f"Total URLs processed: {results}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
