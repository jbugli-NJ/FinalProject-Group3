"""
Tests for src/data_collection/bulk_data.py
Generated with a lot of AI assistance
"""
# %% Imports

import pytest
from unittest.mock import MagicMock, AsyncMock

from data_collection.bulk_data import validate_url


@pytest.mark.asyncio
async def test_validate_url_success():
    """Test validate_url when URL exists (status 200)."""

    mock_session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 200

    mock_session.head.return_value.__aenter__.return_value = mock_response

    test_url = "https://www.govinfo.gov/bulkdata/BILLS/118/1/hr/BILLS-118-1-hr.zip"

    url, status = await validate_url(mock_session, test_url)

    assert url == test_url
    assert status == 200
    mock_session.head.assert_called_once_with(test_url)


@pytest.mark.asyncio
async def test_validate_url_not_found():
    """Test validate_url when URL does not exist (status 404)."""

    mock_session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_session.head.return_value.__aenter__.return_value = mock_response
    test_url = "http://invalid.com/invalid"

    url, status = await validate_url(mock_session, test_url)

    assert url == test_url
    assert status == 404
    mock_session.head.assert_called_once_with(test_url)