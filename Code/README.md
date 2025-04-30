# Project: Legislative Topic Tagging
Author: Jehan Bugli

## Overview
This project aims to automate policy area categorization for congressional bills.
It includes scripts for data collection, training, and inference.

This project uses a src layout, where all the code sits!
The module structure (within src) is as follows:

- data_collection: modules for downloading and processing bulk data
    - bulk_data.py: initial collection steps for bill data
    - parsing.py: retrieving bill text contents for use
    - policy_areas.py: collecting policy area tags for use
    - utils/downloads.py: a download utility used in other files for loading XML



https://www.govinfo.gov/bulkdata/BILLS/resources/billres.xsl

sudo apt install python3.12-venv

python3 -m venv .venv

source .venv/bin/activate (on Linux)

pip install -e .

get-data