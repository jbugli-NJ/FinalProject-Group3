"""
Packages steps converting 'BILLS' bulk data into tabular data for use.
"""

# %% Imports

import os

from bs4 import BeautifulSoup

import polars as pl

from tqdm import tqdm

from data_collection.utils.downloads import load_xml

# %% Define XML parsing workhorse

def xml_bill_to_text(xml_path: str) -> str:
    """
    Converts congressional bill XML into plain text.
    This only retrieves the text body content at <legis-body>.

    :param str xml_path: The .xml file path to reference.

    :return: The bill's contents in plain text.
    """

    xml_content = load_xml(xml_path=xml_path)

    if xml_content is None:
        return None


    try:

        # Get relevant text body with BeautifulSoup

        soup = BeautifulSoup(xml_content, 'xml')

        tags = [
            'preamble',
            'engrossed-amendment-body',
            'resolution-body',
            'legis-body'
        ]

        finds = [soup.find(tag) for tag in tags]

        valid_text_list = [
            f.get_text(separator=' ', strip=True)
            for f in finds
            if f is not None
        ]


        if len(valid_text_list) == 0:
            print(f"No valid tags found: {xml_path}")
            return None
    
        else:
            return '\n\n'.join(valid_text_list)

        
    except Exception as e:

        print(f"Unexpected bill XML parsing error: {e}")
        return None


# %% Define a function to convert an XML file folder to tabular data

def xml_folder_to_dataframe(folder_path: str):
    """
    Converts a folder with .xml bill text files into a Polars DataFrame.
    This parses XML to return the bill's body in plain text.

    :param str folder_path: The path to a folder with .xml files.

    :return: A Polars DataFrame with the following columns:

        - 'file_name': The original .xml file's name
        - 'bill_text': The bill's body in plain text
    """

    bill_text_list = [
        {
            'file_name': file_name,
            'bill_text': xml_bill_to_text(os.path.join(folder_path, file_name))
        }
        for file_name in tqdm(os.listdir(folder_path))
    ]

    df = pl.from_dicts(bill_text_list)

    return df


# %%

if __name__ == '__main__':
    pass