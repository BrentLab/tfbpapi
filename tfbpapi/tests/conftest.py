import pickle
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_cache_info():
    """Load real cache data from pickle file."""
    cache_file = Path(__file__).parent / "data" / "cache_info.pkl"

    if not cache_file.exists():
        pytest.skip(
            "test_cache_data.pkl not found. Run cache data generation script first."
        )

    with open(cache_file, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def mock_scan_cache_dir(mock_cache_info):
    """Mock scan_cache_dir to return our pickled cache data."""
    with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache_info):
        yield mock_cache_info
