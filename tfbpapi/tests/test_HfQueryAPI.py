import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from tfbpapi.HfQueryAPI import HfQueryAPI


@pytest.fixture
def mock_dataset_card():
    """Mock DatasetCard.load to return fake card data."""
    with patch("tfbpapi.HfQueryAPI.DatasetCard.load") as mock:
        mock_card = Mock()
        mock_card.data.to_dict.return_value = {
            "configs": [
                {
                    "config_name": "default",
                    "dataset_info": {
                        "features": [
                            {
                                "name": "text",
                                "dtype": "string",
                                "description": "Input text",
                            },
                            {
                                "name": "label",
                                "dtype": "int64",
                                "description": "Classification label",
                            },
                        ]
                    },
                    "data_files": [
                        {"path": "data/train.parquet", "split": "train"},
                        {"path": "data/test.parquet", "split": "test"},
                    ],
                }
            ]
        }
        mock.return_value = mock_card
        yield mock


@pytest.fixture
def mock_load_dataset():
    """Mock load_dataset to return fake dataset."""
    with patch("tfbpapi.HfQueryAPI.load_dataset") as mock:
        # Create mock dataset with sample data
        mock_dataset_dict = DatasetDict(
            {
                "train": Dataset.from_pandas(
                    pd.DataFrame(
                        {"text": ["hello", "world", "test"], "label": [0, 1, 0]}
                    )
                ),
                "test": Dataset.from_pandas(
                    pd.DataFrame({"text": ["sample", "data"], "label": [1, 0]})
                ),
            }
        )
        mock.return_value = mock_dataset_dict
        yield mock


@pytest.fixture
def mock_duckdb():
    """Mock DuckDB connection."""
    with patch("tfbpapi.HfQueryAPI.duckdb.connect") as mock_connect:
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchdf.return_value = pd.DataFrame({"count": [3]})
        mock_conn.execute.return_value = mock_result
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def mock_hf_api_init():
    """Mock AbstractHfAPI initialization to prevent real API calls."""
    # Instead of mocking the __init__, we'll mock the methods that cause issues
    with patch(
        "tfbpapi.AbstractHfAPI.AbstractHfAPI._get_dataset_size"
    ) as mock_get_size:
        mock_get_size.return_value = None
        yield mock_get_size


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestHfQueryAPI:
    """Test cases for HfQueryAPI."""

    def test_init_with_auto_parse(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test initialization with auto_parse_datacard=True."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Verify initialization
        assert api.auto_download_threshold_mb == 100.0
        assert api._datasets == {
            "default": {
                "features": {
                    "text": {"dtype": "string", "description": "Input text"},
                    "label": {"dtype": "int64", "description": "Classification label"},
                },
                "data_files": [
                    {"path": "data/train.parquet", "split": "train"},
                    {"path": "data/test.parquet", "split": "test"},
                ],
                "config": {
                    "config_name": "default",
                    "dataset_info": {
                        "features": [
                            {
                                "name": "text",
                                "dtype": "string",
                                "description": "Input text",
                            },
                            {
                                "name": "label",
                                "dtype": "int64",
                                "description": "Classification label",
                            },
                        ]
                    },
                    "data_files": [
                        {"path": "data/train.parquet", "split": "train"},
                        {"path": "data/test.parquet", "split": "test"},
                    ],
                },
                "loaded": False,
            }
        }
        mock_dataset_card.assert_called_once()

    def test_init_without_auto_parse(
        self, mock_hf_api_init, mock_duckdb, temp_cache_dir
    ):
        """Test initialization with auto_parse_datacard=False."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=False
        )

        assert api._datasets == {}

    def test_init_parse_datacard_failure(
        self, mock_hf_api_init, mock_duckdb, temp_cache_dir
    ):
        """Test initialization when parse_datacard fails."""
        with patch("tfbpapi.HfQueryAPI.DatasetCard.load") as mock_card:
            mock_card.side_effect = Exception("Failed to load")

            api = HfQueryAPI(
                repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
            )

            assert api._datasets == {}
            # Logger warning should have been called during init

    def test_datasets_property(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test datasets property getter and setter."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Test getter
        assert "default" in api.datasets

        # Test setter
        new_datasets = {"custom": {"features": {}, "data_files": [], "loaded": False}}
        api.datasets = new_datasets
        assert api.datasets == new_datasets

    def test_available_tables(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test available_tables property."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        assert api.available_tables == ["default"]

    def test_parse_datacard_success(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test successful datacard parsing."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=False
        )

        result = api.parse_datacard()

        assert "default" in result
        assert "features" in result["default"]
        assert "text" in result["default"]["features"]
        assert "label" in result["default"]["features"]

    def test_parse_datacard_failure(
        self, mock_hf_api_init, mock_duckdb, temp_cache_dir
    ):
        """Test datacard parsing failure."""
        with patch("tfbpapi.HfQueryAPI.DatasetCard.load") as mock_card:
            mock_card.side_effect = Exception("Load failed")

            api = HfQueryAPI(
                repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=False
            )

            result = api.parse_datacard()

            assert result == {}

    def test_ensure_dataset_loaded_not_found(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test _ensure_dataset_loaded with non-existent table."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            api._ensure_dataset_loaded("nonexistent")

    def test_ensure_dataset_loaded_already_loaded(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test _ensure_dataset_loaded when dataset already loaded."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Pre-load a dataset
        mock_dataset = Mock()
        api._loaded_datasets["default"] = mock_dataset

        result = api._ensure_dataset_loaded("default")
        assert result == mock_dataset
        mock_load_dataset.assert_not_called()

    def test_ensure_dataset_loaded_download_and_load(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test _ensure_dataset_loaded with download and load."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock download method - ensure snapshot_path is None initially
        api.snapshot_path = None
        mock_download = Mock()
        mock_download.return_value = Path("/fake/snapshot")
        api.download = mock_download  # type: ignore

        # Set side effect to simulate download setting snapshot_path
        def download_side_effect(**kwargs):
            api.snapshot_path = Path("/fake/snapshot")
            return Path("/fake/snapshot")

        mock_download.side_effect = download_side_effect

        result = api._ensure_dataset_loaded("default")

        assert result == mock_load_dataset.return_value
        mock_download.assert_called_once_with(auto_download_threshold_mb=100.0)
        mock_load_dataset.assert_called_once()
        assert api._datasets["default"]["loaded"] is True

    def test_query_with_table_name(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test query with explicit table name."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock the dataset loading
        api.snapshot_path = Path("/fake/snapshot")

        result = api.query("SELECT * FROM default", table_name="default")

        assert isinstance(result, pd.DataFrame)
        mock_duckdb.execute.assert_called_once_with("SELECT * FROM default")

    def test_query_infer_table_name(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test query with table name inference."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock the dataset loading
        api.snapshot_path = Path("/fake/snapshot")

        result = api.query("SELECT * FROM default")

        assert isinstance(result, pd.DataFrame)
        mock_duckdb.execute.assert_called_once_with("SELECT * FROM default")

    def test_describe_table(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test describe_table method."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock the dataset loading
        api.snapshot_path = Path("/fake/snapshot")

        result = api.describe_table("default")

        assert isinstance(result, pd.DataFrame)
        mock_duckdb.execute.assert_called_with("DESCRIBE default")

    def test_sample(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test sample method."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock the dataset loading
        api.snapshot_path = Path("/fake/snapshot")

        result = api.sample("default", n=3)

        assert isinstance(result, pd.DataFrame)
        mock_duckdb.execute.assert_called_with("SELECT * FROM default LIMIT 3")

    def test_count(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test count method."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock the dataset loading
        api.snapshot_path = Path("/fake/snapshot")

        result = api.count("default")

        assert result == 3
        mock_duckdb.execute.assert_called_with("SELECT COUNT(*) as count FROM default")

    def test_get_columns(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test get_columns method."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        result = api.get_columns("default")
        assert result == ["text", "label"]

    def test_get_columns_not_found(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test get_columns with non-existent table."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            api.get_columns("nonexistent")

    def test_context_manager(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test context manager functionality."""
        with HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        ) as api:
            assert api is not None

        # Verify close was called
        mock_duckdb.close.assert_called_once()

    def test_close(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test close method."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        api.close()
        mock_duckdb.close.assert_called_once()

    def test_table_filters_basic(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test basic table filter functionality."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Test setting and getting filters
        assert api.get_table_filter("default") is None

        api.set_table_filter("default", "text = 'test'")
        assert api.get_table_filter("default") == "text = 'test'"

        # Test removing filters
        api.remove_table_filter("default")
        assert api.get_table_filter("default") is None

    def test_table_filters_query_modification(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test that table filters modify queries correctly."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock the dataset loading
        api.snapshot_path = Path("/fake/snapshot")

        # Set a filter
        api.set_table_filter("default", "label = 1")

        # Execute a query
        api.query("SELECT * FROM default", table_name="default")

        # Verify the query was modified to include the filter
        expected_sql = "SELECT * FROM (SELECT * FROM default WHERE label = 1)"
        mock_duckdb.execute.assert_called_once_with(expected_sql)

    def test_table_filters_no_modification_when_no_filters(
        self,
        mock_hf_api_init,
        mock_dataset_card,
        mock_load_dataset,
        mock_duckdb,
        temp_cache_dir,
    ):
        """Test that queries are not modified when no filters are set."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Mock the dataset loading
        api.snapshot_path = Path("/fake/snapshot")

        # Execute a query without any filters
        original_sql = "SELECT * FROM default"
        api.query(original_sql, table_name="default")

        # Verify the query was not modified
        mock_duckdb.execute.assert_called_once_with(original_sql)

    def test_apply_table_filters_method(
        self, mock_hf_api_init, mock_dataset_card, mock_duckdb, temp_cache_dir
    ):
        """Test the _apply_table_filters method directly."""
        api = HfQueryAPI(
            repo_id="test/repo", cache_dir=temp_cache_dir, auto_parse_datacard=True
        )

        # Test with no filters
        sql = "SELECT * FROM default"
        assert api._apply_table_filters(sql) == sql

        # Test with filter applied
        api.set_table_filter("default", "text LIKE '%test%'")
        modified_sql = api._apply_table_filters(sql)
        expected = "SELECT * FROM (SELECT * FROM default WHERE text LIKE '%test%')"
        assert modified_sql == expected

        # Test that already filtered queries don't get double-wrapped
        already_filtered = "SELECT * FROM (SELECT * FROM default WHERE existing = 1)"
        result = api._apply_table_filters(already_filtered)
        # Should not modify since it already contains a filtered subquery
        assert result == already_filtered
