import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from requests import HTTPError

from tfbpapi.AbstractHfAPI import AbstractHfAPI, RepoTooLargeError


class TestHfAPI(AbstractHfAPI):
    """Concrete implementation of AbstractHfAPI for testing."""

    def parse_datacard(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Test implementation of parse_datacard."""
        return {"test": "datacard"}

    def query(self, *args: Any, **kwargs: Any) -> Any:
        """Test implementation of query."""
        return {"test": "query"}


@pytest.fixture
def mock_hf_hub_download():
    """Mock hf_hub_download to return a fake path."""
    with patch("tfbpapi.AbstractHfAPI.hf_hub_download") as mock:
        mock.return_value = "/fake/path/to/file.txt"
        yield mock


@pytest.fixture
def mock_snapshot_download():
    """Mock snapshot_download to return a fake path."""
    with patch("tfbpapi.AbstractHfAPI.snapshot_download") as mock:
        mock.return_value = "/fake/path/to/snapshot"
        yield mock


@pytest.fixture
def mock_repo_info():
    """Mock repo_info to return fake repo information."""
    with patch("tfbpapi.AbstractHfAPI.repo_info") as mock:
        # Create a mock with siblings attribute
        mock_info = Mock()
        mock_info.siblings = [
            Mock(size=1024 * 1024),  # 1MB file
            Mock(size=512 * 1024),  # 512KB file
            Mock(size=None),  # File with no size
        ]
        mock.return_value = mock_info
        yield mock


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for dataset size API calls."""
    with patch("tfbpapi.AbstractHfAPI.requests.get") as mock:
        # Create a mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "size": {
                "dataset": {
                    "num_bytes_original_files": 10 * 1024 * 1024,  # 10MB
                    "size_determination_complete": True,
                }
            },
            "partial": False,
        }
        mock_response.raise_for_status.return_value = None
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def mock_dataset_size_call():
    """Mock the _get_dataset_size call to prevent real API calls during init."""
    with patch.object(AbstractHfAPI, "_get_dataset_size") as mock:
        yield mock


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestAbstractHfAPI:
    """Test cases for AbstractHfAPI."""

    def test_init_basic(self, temp_cache_dir, mock_dataset_size_call):
        """Test basic initialization."""
        api = TestHfAPI(
            repo_id="test/repo",
            repo_type="dataset",
            token="test-token",
            cache_dir=temp_cache_dir,
        )

        assert api.repo_id == "test/repo"
        assert api.repo_type == "dataset"
        assert api.token == "test-token"
        assert api.cache_dir == temp_cache_dir

    def test_init_with_env_vars(
        self, temp_cache_dir, monkeypatch, mock_dataset_size_call
    ):
        """Test initialization with environment variables."""
        monkeypatch.setenv("HF_TOKEN", "env-token")
        monkeypatch.setenv("HF_CACHE_DIR", str(temp_cache_dir))

        api = TestHfAPI(repo_id="test/repo")

        assert api.token == "env-token"
        assert api.cache_dir == temp_cache_dir

    def test_init_user_overrides_env(self, temp_cache_dir, monkeypatch):
        """Test that user parameters override environment variables."""
        monkeypatch.setenv("HF_TOKEN", "env-token")

        api = TestHfAPI(
            repo_id="test/repo", token="user-token", cache_dir=temp_cache_dir
        )

        assert api.token == "user-token"
        assert api.cache_dir == temp_cache_dir

    def test_cache_dir_setter_valid(self, temp_cache_dir):
        """Test cache_dir setter with valid directory."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        new_cache_dir = temp_cache_dir / "new_cache"
        new_cache_dir.mkdir()

        api.cache_dir = new_cache_dir
        assert api.cache_dir == new_cache_dir

    def test_cache_dir_setter_invalid(self, temp_cache_dir):
        """Test cache_dir setter with invalid directory."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        invalid_dir = temp_cache_dir / "nonexistent"

        with pytest.raises(
            FileNotFoundError, match="Cache directory .* does not exist"
        ):
            api.cache_dir = invalid_dir

    @patch("tfbpapi.AbstractHfAPI.requests.get")
    def test_get_dataset_size_success(self, mock_get, temp_cache_dir):
        """Test successful dataset size retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "size": {"dataset": {"num_bytes": 1024}},
            "partial": False,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api._get_dataset_size()

        assert api.size is not None
        assert not api.size.get("partial", True)

    @patch("tfbpapi.AbstractHfAPI.requests.get")
    def test_get_dataset_size_partial(self, mock_get, temp_cache_dir):
        """Test dataset size retrieval with partial results."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "size": {"dataset": {"num_bytes": 1024}},
            "partial": True,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api._get_dataset_size()

        assert api.size["partial"] is True  # type: ignore[index]
        assert "size_warning" in api.size["size"]["dataset"]  # type: ignore[index]

    def test_repo_id_setter_success(self, mock_requests_get, temp_cache_dir):
        """Test successful repo_id setting."""
        api = TestHfAPI(repo_id="initial/repo", cache_dir=temp_cache_dir)

        api.repo_id = "new/repo"
        assert api.repo_id == "new/repo"
        mock_requests_get.assert_called()

    @patch("tfbpapi.AbstractHfAPI.requests.get")
    def test_repo_id_setter_failure(self, mock_get, temp_cache_dir):
        """Test repo_id setting with API failure."""
        mock_get.side_effect = HTTPError("Repository not found")

        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        # Should not raise, but should log error
        api.repo_id = "nonexistent/repo"
        assert api.repo_id == "nonexistent/repo"

    def test_get_dataset_size_mb(self, temp_cache_dir):
        """Test dataset size calculation in MB."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        # Test with no size data
        assert api._get_dataset_size_mb() == float("inf")

        # Test with size data
        api.size = {
            "size": {"dataset": {"num_bytes_original_files": 2 * 1024 * 1024}}  # 2MB
        }
        assert api._get_dataset_size_mb() == 2.0

    def test_build_auth_headers(self, temp_cache_dir):
        """Test authentication header building."""
        # Without token
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api.token = None
        assert api._build_auth_headers() == {}

        # With token
        api.token = "test-token"
        headers = api._build_auth_headers()
        assert headers == {"Authorization": "Bearer test-token"}

    def test_ensure_str_paths(self, temp_cache_dir):
        """Test path string conversion."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        kwargs = {
            "local_dir": Path("/some/path"),
            "cache_dir": Path("/cache/path"),
            "other_param": "unchanged",
        }

        api._ensure_str_paths(kwargs)

        assert kwargs["local_dir"] == "/some/path"
        assert kwargs["cache_dir"] == "/cache/path"
        assert kwargs["other_param"] == "unchanged"

    def test_normalize_patterns(self, temp_cache_dir):
        """Test pattern normalization."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        kwargs = {
            "allow_patterns": "*.txt",
            "ignore_patterns": ["*.log", "*.tmp"],
            "other_param": "unchanged",
        }

        api._normalize_patterns(kwargs)

        assert kwargs["allow_patterns"] == ["*.txt"]
        assert kwargs["ignore_patterns"] == ["*.log", "*.tmp"]
        assert kwargs["other_param"] == "unchanged"

    def test_download_single_file(self, mock_hf_hub_download, temp_cache_dir):
        """Test single file download."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        result = api._download_single_file("test.txt")

        assert result == Path("/fake/path/to/file.txt")
        mock_hf_hub_download.assert_called_once()

        # Check that correct arguments were passed
        call_args = mock_hf_hub_download.call_args[1]
        assert call_args["repo_id"] == "test/repo"
        assert call_args["filename"] == "test.txt"

    def test_download_single_file_dry_run(self, mock_hf_hub_download, temp_cache_dir):
        """Test single file download with dry run."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        result = api._download_single_file("test.txt", dry_run=True)

        assert result == Path("dry_run_path")
        mock_hf_hub_download.assert_not_called()

    def test_download_snapshot(self, mock_snapshot_download, temp_cache_dir):
        """Test snapshot download."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        result = api._download_snapshot(allow_patterns=["*.txt"])

        assert result == Path("/fake/path/to/snapshot")
        mock_snapshot_download.assert_called_once()

    def test_download_snapshot_dry_run(self, mock_snapshot_download, temp_cache_dir):
        """Test snapshot download with dry run."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        result = api._download_snapshot(dry_run=True, allow_patterns=["*.txt"])

        assert result == Path("dry_run_path")
        mock_snapshot_download.assert_not_called()

    def test_download_single_file_string(self, temp_cache_dir):
        """Test download with single file as string."""
        # Create API instance by bypassing problematic initialization
        api = TestHfAPI.__new__(TestHfAPI)  # Create without calling __init__

        # Manually set the required attributes
        api._repo_id = "test/repo"
        api.repo_type = "dataset"
        api.token = None
        api._cache_dir = temp_cache_dir
        api.size = {
            "size": {"dataset": {"num_bytes_original_files": 1024}}
        }  # Small size
        api.logger = Mock()  # Mock logger to avoid issues

        with patch("tfbpapi.AbstractHfAPI.hf_hub_download") as mock_hf_download:
            # Configure the mock to return a fake path
            mock_hf_download.return_value = "/fake/path/to/file.txt"

            result = api.download(files="test.txt", auto_download_threshold_mb=0)

            assert result == Path("/fake/path/to/file.txt")
            mock_hf_download.assert_called_once()

            # Verify the call arguments
            call_args = mock_hf_download.call_args[1]
            assert call_args["repo_id"] == "test/repo"
            assert call_args["filename"] == "test.txt"

    def test_download_single_file_list(
        self, mock_hf_hub_download, temp_cache_dir, mock_dataset_size_call
    ):
        """Test download with single file as list."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api.size = {
            "size": {"dataset": {"num_bytes_original_files": 1024}}
        }  # Small size

        result = api.download(files=["test.txt"], auto_download_threshold_mb=0)

        assert result == Path("/fake/path/to/file.txt")
        mock_hf_hub_download.assert_called_once()

    def test_download_multiple_files(self, mock_snapshot_download, temp_cache_dir):
        """Test download with multiple files."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api.size = {
            "size": {"dataset": {"num_bytes_original_files": 1024}}
        }  # Small size

        result = api.download(files=["test1.txt", "test2.txt"])

        assert result == Path("/fake/path/to/snapshot")
        mock_snapshot_download.assert_called_once()

    def test_download_force_full(self, mock_snapshot_download, temp_cache_dir):
        """Test download with force_full_download=True."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api.size = {
            "size": {"dataset": {"num_bytes_original_files": 1000 * 1024 * 1024}}
        }  # Large size

        result = api.download(force_full_download=True)

        assert result == Path("/fake/path/to/snapshot")
        mock_snapshot_download.assert_called_once()

    def test_download_repo_too_large(self, temp_cache_dir):
        """Test download with repo too large error."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api.size = {
            "size": {"dataset": {"num_bytes_original_files": 1000 * 1024 * 1024}}
        }  # 1GB

        with pytest.raises(RepoTooLargeError, match="Dataset size .* exceeds"):
            api.download(auto_download_threshold_mb=10)

    def test_download_small_repo_auto(self, mock_snapshot_download, temp_cache_dir):
        """Test download with small repo under threshold."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)
        api.size = {
            "size": {"dataset": {"num_bytes_original_files": 5 * 1024 * 1024}}
        }  # 5MB

        result = api.download(auto_download_threshold_mb=10)

        assert result == Path("/fake/path/to/snapshot")
        mock_snapshot_download.assert_called_once()

    def test_snapshot_path_property(self, temp_cache_dir):
        """Test snapshot_path property getter and setter."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        # Initially None
        assert api.snapshot_path is None

        # Set to path
        test_path = "/some/path"
        api.snapshot_path = Path(test_path)
        assert api.snapshot_path == Path(test_path)

        # Set to None
        api.snapshot_path = None
        assert api.snapshot_path is None

    def test_size_property(self, temp_cache_dir):
        """Test size property getter and setter."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        # Initially None
        assert api.size is None

        # Set size data
        size_data = {"size": {"dataset": {"num_bytes": 1024}}}
        api.size = size_data
        assert api.size == size_data

    def test_abstract_methods_implemented(self, temp_cache_dir):
        """Test that abstract methods are properly implemented."""
        api = TestHfAPI(repo_id="test/repo", cache_dir=temp_cache_dir)

        # Test parse_datacard
        result = api.parse_datacard()
        assert result == {"test": "datacard"}

        # Test query
        result = api.query()
        assert result == {"test": "query"}

    def test_repo_too_large_error(self):
        """Test RepoTooLargeError exception."""
        error = RepoTooLargeError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, ValueError)
