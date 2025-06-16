import pandas as pd
import os

# path to data folder
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))


def extract_science_category() -> None:
    """
    Extracts entries belonging to the science category from a JSON dataset and saves
    the filtered data into a new JSON lines file.

    This function reads a JSON dataset containing categories of news, filters news entries
    that belong specifically to the "SCIENCE" category, removes corrupted links, and limits
    the records to dates starting from 2015. The filtered data is formatted into JSON lines
    with properly formatted dates, and the output is written to a new file.

    :return: None
    :rtype: None
    """
    source_file = os.path.join(DATA_PATH, "News_Category_Dataset_v3.json")
    output_file = os.path.join(DATA_PATH, "science_category.jsonl")

    data_df = pd.read_json(source_file, lines=True)
    # science category is selected only and corrupted links are removed
    condition = (
        (data_df["category"] == "SCIENCE") &
        (~data_df["link"].str.contains(r"http.+https?")) &
        (data_df["date"] >= "2015-01-01")
    )
    data_science_df = data_df.loc[condition, ["link", "headline", "date"]]
    data_science_df['date'] = pd.to_datetime(data_science_df['date']).dt.strftime('%Y-%m-%d')

    formatted_json = data_science_df.to_json(orient='records', lines=True, date_format="iso").replace('\\/', '/')
    print(formatted_json, file=open(output_file, 'w'))
