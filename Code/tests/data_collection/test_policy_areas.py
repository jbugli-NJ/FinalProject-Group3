"""
Tests for src/data_collection/policy_areas.py
Generated with a lot of AI assistance
"""
# %% Imports

import pytest

from data_collection.policy_areas import extract_file_name_from_url


@pytest.mark.parametrize("url, expected_filename", [
    ("https://www.govinfo.gov/content/pkg/BILLS-118hr5784ih/xml/BILLS-118hr5784ih.xml", "BILLS-118hr5784ih.xml"),
    ("https://www.govinfo.gov/content/pkg/BILLS-118hr2pcs/xml/BILLS-118hr2pcs.xml", "BILLS-118hr2pcs.xml"),
])
def test_extract_file_name_from_url_success(url, expected_filename):
    """Test successful extraction of filenames from various URL formats."""
    assert extract_file_name_from_url(url) == expected_filename