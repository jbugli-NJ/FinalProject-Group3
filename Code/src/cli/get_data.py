"""
Chains steps to:

1. Download and unzip GovInfo bulk data
2. Parse XML files for plain text
3. Convert to tabular data for training
"""
# %% Imports

import asyncio

import aiohttp

import polars as pl

from data_collection.bulk_data import (
    construct_bill_download_urls,
    download_text,
    unzip_all
)

from data_collection.parsing import xml_folder_to_dataframe

from data_collection.policy_areas import get_all_policy_areas

# %% Define main executor function

# NOTE: uses an internal main() to allow for CLI script execution

async def _main():
    """
    Executes steps to get tabular data from GovInfo bulk data files.
    """

    # Download bulk data

    connector = aiohttp.TCPConnector(limit_per_host=5)
    timeout = aiohttp.ClientTimeout(600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        text_urls = await construct_bill_download_urls(
            session=session,
            congresses=range(113, 120, 1),
            data_type='BILLS'
        )

        text_paths = await download_text(session=session,
            urls=text_urls,
            zipped_dir='zipped_bills'
        )

        status_urls = await construct_bill_download_urls(
            session=session,
            congresses=range(113, 120, 1),
            data_type='BILLSTATUS'
        )

        status_paths = await download_text(
            session=session,
            urls=status_urls,
            zipped_dir='zipped_status'
        )


    # Unzip files to separate folders

    print('Unzipping data to flatten folders ...')

    unzip_all(zip_paths=text_paths, unzipped_dir='unzipped_bills')
    unzip_all(zip_paths=status_paths, unzipped_dir='unzipped_status')


    # Build dataframes from unzipped bill text files and other data

    text_df = xml_folder_to_dataframe('unzipped_bills')
    text_df.write_parquet('parsed_bill_text.parquet')

    area_df = get_all_policy_areas('unzipped_status')
    area_df.write_parquet('policy_areas.parquet')


    # Combine dataframes to get training data

    combined_df = pl.read_parquet('policy_areas.parquet').join(
        pl.read_parquet('parsed_bill_text.parquet'),
        left_on='bill_text_file_name',
        right_on='file_name',
        how='inner'
    )

    combined_df.write_parquet('input_data.parquet')


def main():
    asyncio.run(_main())

if __name__ == '__main__':
    main()
