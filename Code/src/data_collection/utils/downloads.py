"""
Download utilities for use in data collection.
"""

# %% Imports

import os

from typing import Optional

# %% Main XML download script

def load_xml(xml_path: str) -> Optional[str]:
    """
    Loads an .xml file from a given path, returning the contents.

    :param str xml_path: The .xml file path to reference.

    :return: The XML contents or None.
    """

    path_exists = os.path.exists(xml_path)

    if path_exists == False:
        print(f"Path not found: {xml_path}")
        return None

    try:

        with open(xml_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()

        return xml_content

    except Exception as e:

        print(f"Error loading {xml_path}: {e}")

        return None