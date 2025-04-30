"""
Collects existing policy area tags from 'BILLSTATUS' bulk data.
"""

# %% Imports

import os

import re

from bs4 import BeautifulSoup

import polars as pl

from tqdm import tqdm

from typing import Optional

from data_collection.utils.downloads import load_xml


# %% Define helper function for URL processing

def extract_file_name_from_url(url: str) -> Optional[str]:
    """
    Extracts the last part of a govinfo bill URL starting with 'BILLS-'.

    :param str url: The URL string.

    :return: The final BILLS-... section matching downloaded file names.
    """

    match = re.search(r'BILLS-[^/]+$', url)

    return match.group(0) if match is not None else None


# %% Define policy area retrieval function

def get_policy_area(xml_path: str) -> str:
    """
    Retrieves a CRS-assigned policy area tag from bill data XML.
    This also returns associated bill text version files for matching.

    :param str xml_path: The .xml file path to reference.

    :return: A list of dictionaries with the following fields:
    
        - 'policy_area': The bill's policy area
        - 'bill_text_file_name': The associated bill text file name
    """

    xml_content = load_xml(xml_path=xml_path)

    if xml_content is None:
        return []


    try:

        # Get relevant fields with BeautifulSoup

        soup = BeautifulSoup(xml_content, 'xml')

        area_content = soup.find('policyArea')
        url_content = soup.find('textVersions')


        # If both aren't present, return nothing

        if area_content is None or url_content is None:
            return []
        

        # Return a list of dictionaries for the policy area + text version

        policy_area = area_content.get_text(strip=True)

        outputs = [
            {
                'policy_area': policy_area,
                'bill_text_file_name': extract_file_name_from_url(
                    url_tag.get_text(strip=True)
                )
            }
            for url_tag in url_content.find_all('url')
        ]

        return outputs

        
    except Exception as e:

        print(f"Unexpected error for {xml_path}: {e}")
        return []


# %% Define a function to convert an XML file folder to tabular data

def get_all_policy_areas(folder_path: str):
    """
    Converts a folder with .xml bill data files into a Polars DataFrame.
    This retrieves policy area tags and associated bill text file names.

    :param str folder_path: The path to a folder with .xml files.

    :return: A Polars DataFrame with the following columns:
    """

    policy_area_list = []

    for file_name in tqdm(os.listdir(folder_path)):

        area_items = get_policy_area(
            os.path.join(folder_path, file_name)
        )

        policy_area_list.extend(area_items)


    df = pl.from_dicts(policy_area_list)

    return df


# %%

if __name__ == '__main__':
    df = get_all_policy_areas('unzipped_status')
    df.write_parquet('policy_areas.parquet')