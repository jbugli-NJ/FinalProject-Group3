# %% Imports

import asyncio

import aiohttp

import os

import random

# %% Specify static inputs to use for URL construction

bill_types = [
    'hr',
    's',
    'hres',
    'sres',
    'hjres',
    'sjres',
    'hconres',
    'sconres'
]

# %% Define function to generate download links

async def construct_urls(
    session: aiohttp.ClientSession,
    congresses: list[int]
    ):

    """
    Cycle through available bill types and congresses to construct links for use.

    :param aiohttp.ClientSession session: An aiohttp client session to use for downloads.

    :param list[int] congresses: The list of available congressional sessions to use.

    :return list[str]: A list of validated .zip download links.
    """

    async def validate_url(url: str) -> bool:
        """
        Validates a URL through a HEAD request to make sure it exists.
        As of writing this, the 119th Congress doesn't have a second session.

        :param str url: The URL to test.

        :return: True if the URL exists and False otherwise.
        """

        async with session.head(url) as response:
            return (url, response.status)

    raw_urls = [
        f"https://www.govinfo.gov/bulkdata/BILLS/{c}/{s}/{t}/BILLS-{c}-{s}-{t}.zip"
        for c in congresses
        for s in [1,2]
        for t in bill_types
    ]

    tasks = [
        asyncio.create_task(validate_url(raw_url))
        for raw_url in raw_urls
    ]

    validation_results = await asyncio.gather(*tasks)

    valid_urls = [u for u, v in validation_results if v == 200]
    invalid_urls = [u for u, v in validation_results if v != 200]

    if len(invalid_urls) > 0:
        print(f"Invalid URLs (119th Congress Session 2 expected):\n{'\n'.join(invalid_urls)}\n\n")

    return valid_urls


# %%

async def download_bulk_data_file(
    session: aiohttp.ClientSession,
    url: str,
    save_path: str
    ):

    """
    Downloads bulk data from a given URL as a .zip file.
    Implements retry logic depending on the error tyle.
   
    :param aiohttp.ClientSession: An aiohttp client session to use for downloads.

    :param str url: A valid .zip download link.

    :param str save_path: The full path to use for saving the zipped contents, including the '.zip' extension.

    :return: The path to the saved file, or None upon failure.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    await asyncio.sleep(random.uniform(3,10))

    # Define steps to download and save the zip file
    # If successful, the zip file is saved; otherwise, the function retries or exits

    for attempt in range(1, 4):

        try:

            async with session.get(url) as response:

                if response.status == 200:
                    print(f"SUCCESS: {url}")
                    zip_content = await response.read()

                    with open(save_path, 'wb') as file:
                        file.write(zip_content)

                    return save_path
                
                else:

                    print(f"ERROR ({response.status}): {url}")

                    if response.status == 404:
                        print(f"FAILED: {url}")
                        return None

                    
                response.raise_for_status()
                
        except (
            aiohttp.ClientResponseError,
            aiohttp.ClientPayloadError,
            aiohttp.ConnectionTimeoutError,
            ):

            if attempt == 3:
                print(f"FAILED: {url}")
                return None
            
            else:
                print(f"Attempting retry ({attempt}): {url}")
                await asyncio.sleep(attempt*5)


# %% Define bulk data download function

async def download_all_data(
    session: aiohttp.ClientSession,
    urls: list[str]
    ) -> list[str]:

    """
    Downloads a list of bulk data .zip files using the provided URLs.

    :param aiohttp.ClientSession session: An aiohttp client session to use for downloads.

    :param list[str] urls: A list of URLs to download.

    :return: Paths to the downloaded .zip files.
    """

    tasks = [
        asyncio.create_task(
            download_bulk_data_file(
            session=session,
            url=url,
            save_path=os.path.join(os.getcwd(), 'zips', url.rsplit(r'/', 1)[-1])
            )
        )
        for url in urls
    ]

    raw_paths = await asyncio.gather(*tasks, return_exceptions=True)

    valid_paths = [r for r in raw_paths if not isinstance(r, Exception) and r is not None]
    invalid_paths = [r for r in raw_paths if isinstance(r, Exception) or r is not None]

    if len(invalid_paths) > 0:

        print(f"The following expected downloads failed:\n{'\n'.join(invalid_paths)}\n\n")

    return valid_paths


# %%

async def main():

    connector = aiohttp.TCPConnector(limit_per_host=5)
    timeout = aiohttp.ClientTimeout(150)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        urls = await construct_urls(session=session, congresses=range(113, 120, 1))

        paths = await download_all_data(session=session, urls=urls)

        return paths

if __name__ == '__main__':
    print(asyncio.run(main()))
