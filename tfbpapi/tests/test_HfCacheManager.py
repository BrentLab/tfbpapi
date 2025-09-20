"""Comprehensive tests for HfCacheManager class."""

import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import duckdb
import pytest

from tfbpapi.datainfo.models import DatasetType
from tfbpapi.HfCacheManager import HfCacheManager


class TestHfCacheManagerInit:
    """Test HfCacheManager initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        conn = duckdb.connect(":memory:")
        repo_id = "test/repo"

        with patch(
            "tfbpapi.HfCacheManager.DataCard.__init__", return_value=None
        ) as mock_datacard_init:
            cache_manager = HfCacheManager(repo_id, conn)
            # Manually set the properties that would normally
            # be set by DataCard.__init__
            cache_manager.repo_id = repo_id
            cache_manager.token = None

            assert cache_manager.repo_id == repo_id
            assert cache_manager.duckdb_conn == conn
            assert cache_manager.token is None
            assert cache_manager.logger is not None
            # DataCard should be initialized as parent
            mock_datacard_init.assert_called_once_with(repo_id, None)

    def test_init_with_token_and_logger(self):
        """Test initialization with token and custom logger."""
        conn = duckdb.connect(":memory:")
        repo_id = "test/repo"
        token = "test_token"
        logger = logging.getLogger("test_logger")

        with patch(
            "tfbpapi.HfCacheManager.DataCard.__init__", return_value=None
        ) as mock_datacard_init:
            cache_manager = HfCacheManager(repo_id, conn, token=token, logger=logger)
            # Manually set the properties that would
            # normally be set by DataCard.__init__
            cache_manager.repo_id = repo_id
            cache_manager.token = token

            assert cache_manager.repo_id == repo_id
            assert cache_manager.duckdb_conn == conn
            assert cache_manager.token == token
            assert cache_manager.logger == logger
            # DataCard should be initialized as parent with token
            mock_datacard_init.assert_called_once_with(repo_id, token)


class TestHfCacheManagerDatacard:
    """Test DataCard integration since HfCacheManager now inherits from DataCard."""

    def test_datacard_inheritance(self):
        """Test that HfCacheManager properly inherits from DataCard."""
        conn = duckdb.connect(":memory:")
        repo_id = "test/repo"
        token = "test_token"

        with patch(
            "tfbpapi.HfCacheManager.DataCard.__init__", return_value=None
        ) as mock_datacard_init:
            cache_manager = HfCacheManager(repo_id, conn, token=token)

            # DataCard should be initialized during construction
            mock_datacard_init.assert_called_once_with(repo_id, token)

            # Should have DataCard methods available (they exist on the class)
            assert hasattr(cache_manager, "get_config")
            assert hasattr(cache_manager, "get_configs_by_type")


class TestHfCacheManagerDuckDBOperations:
    """Test DuckDB operations that are still part of HfCacheManager."""

    @patch("tfbpapi.HfCacheManager.DataCard.__init__", return_value=None)
    def test_create_duckdb_table_from_files_single_file(
        self, mock_datacard_init, tmpdir
    ):
        """Test creating DuckDB table from single parquet file."""
        # Create a mock parquet file
        parquet_file = tmpdir.join("test.parquet")
        parquet_file.write("dummy_content")

        # Use a separate cache manager with mock connection for this test
        mock_conn = Mock()
        test_cache_manager = HfCacheManager("test/repo", mock_conn)

        test_cache_manager._create_duckdb_table_from_files(
            [str(parquet_file)], "test_table"
        )

        mock_conn.execute.assert_called_once()
        sql_call = mock_conn.execute.call_args[0][0]
        assert "CREATE OR REPLACE VIEW test_table" in sql_call
        assert str(parquet_file) in sql_call

    @patch("tfbpapi.HfCacheManager.DataCard.__init__", return_value=None)
    def test_create_duckdb_table_from_files_multiple_files(
        self, mock_datacard_init, tmpdir
    ):
        """Test creating DuckDB table from multiple parquet files."""
        # Create mock parquet files
        file1 = tmpdir.join("test1.parquet")
        file1.write("dummy_content1")
        file2 = tmpdir.join("test2.parquet")
        file2.write("dummy_content2")

        files = [str(file1), str(file2)]

        # Use a separate cache manager with mock connection for this test
        mock_conn = Mock()
        test_cache_manager = HfCacheManager("test/repo", mock_conn)

        test_cache_manager._create_duckdb_table_from_files(files, "test_table")

        mock_conn.execute.assert_called_once()
        sql_call = mock_conn.execute.call_args[0][0]
        assert "CREATE OR REPLACE VIEW test_table" in sql_call
        assert str(file1) in sql_call
        assert str(file2) in sql_call


class TestHfCacheManagerCacheManagement:
    """Test cache management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("tfbpapi.HfCacheManager.DataCard.__init__", return_value=None):
            self.conn = duckdb.connect(":memory:")
            self.repo_id = "test/repo"
            self.cache_manager = HfCacheManager(self.repo_id, self.conn)

    def test_parse_size_string(self):
        """Test size string parsing."""
        assert self.cache_manager._parse_size_string("10KB") == 10 * 1024
        assert self.cache_manager._parse_size_string("5MB") == 5 * 1024**2
        assert self.cache_manager._parse_size_string("2GB") == 2 * 1024**3
        assert self.cache_manager._parse_size_string("1TB") == 1 * 1024**4
        assert self.cache_manager._parse_size_string("500") == 500
        assert self.cache_manager._parse_size_string("10.5GB") == int(10.5 * 1024**3)

    def test_format_bytes(self):
        """Test byte formatting."""
        assert self.cache_manager._format_bytes(0) == "0B"
        assert self.cache_manager._format_bytes(1023) == "1023.0B"
        assert self.cache_manager._format_bytes(1024) == "1.0KB"
        assert self.cache_manager._format_bytes(1024**2) == "1.0MB"
        assert self.cache_manager._format_bytes(1024**3) == "1.0GB"
        assert self.cache_manager._format_bytes(1024**4) == "1.0TB"

    @patch("tfbpapi.HfCacheManager.scan_cache_dir")
    def test_clean_cache_by_age(self, mock_scan_cache_dir):
        """Test age-based cache cleaning."""
        # Setup mock cache info
        mock_cache_info = Mock()
        mock_revision = Mock()
        mock_revision.commit_hash = "abc123"
        mock_revision.last_modified = (datetime.now() - timedelta(days=35)).timestamp()

        mock_repo = Mock()
        mock_repo.revisions = [mock_revision]

        mock_cache_info.repos = [mock_repo]
        mock_delete_strategy = Mock()
        mock_delete_strategy.expected_freed_size_str = "100MB"
        mock_cache_info.delete_revisions.return_value = mock_delete_strategy

        mock_scan_cache_dir.return_value = mock_cache_info

        result = self.cache_manager.clean_cache_by_age(max_age_days=30, dry_run=True)

        assert result == mock_delete_strategy
        mock_cache_info.delete_revisions.assert_called_once_with("abc123")

    @patch("tfbpapi.HfCacheManager.scan_cache_dir")
    def test_clean_cache_by_age_no_old_revisions(self, mock_scan_cache_dir):
        """Test age-based cleaning when no old revisions exist."""
        mock_cache_info = Mock()
        mock_revision = Mock()
        mock_revision.commit_hash = "abc123"
        mock_revision.last_modified = datetime.now().timestamp()  # Recent

        mock_repo = Mock()
        mock_repo.revisions = [mock_revision]

        mock_cache_info.repos = [mock_repo]
        mock_delete_strategy = Mock()
        mock_delete_strategy.expected_freed_size_str = "0B"
        mock_cache_info.delete_revisions.return_value = mock_delete_strategy

        mock_scan_cache_dir.return_value = mock_cache_info

        result = self.cache_manager.clean_cache_by_age(max_age_days=30, dry_run=True)

        # Should still return a strategy, but with empty revisions
        assert result == mock_delete_strategy
        mock_cache_info.delete_revisions.assert_called_once_with()

    @patch("tfbpapi.HfCacheManager.scan_cache_dir")
    def test_clean_cache_by_size(self, mock_scan_cache_dir):
        """Test size-based cache cleaning."""
        # Setup mock cache info
        mock_cache_info = Mock()
        mock_cache_info.size_on_disk = 5 * 1024**3  # 5GB
        mock_cache_info.size_on_disk_str = "5.0GB"

        mock_revision = Mock()
        mock_revision.commit_hash = "abc123"
        mock_revision.last_modified = datetime.now().timestamp()
        mock_revision.size_on_disk = 2 * 1024**3  # 2GB

        mock_repo = Mock()
        mock_repo.revisions = [mock_revision]

        mock_cache_info.repos = [mock_repo]
        mock_delete_strategy = Mock()
        mock_delete_strategy.expected_freed_size_str = "2GB"
        mock_cache_info.delete_revisions.return_value = mock_delete_strategy

        mock_scan_cache_dir.return_value = mock_cache_info

        result = self.cache_manager.clean_cache_by_size(
            target_size="3GB", strategy="oldest_first", dry_run=True
        )

        assert result == mock_delete_strategy
        mock_cache_info.delete_revisions.assert_called_once()

    @patch("tfbpapi.HfCacheManager.scan_cache_dir")
    def test_clean_cache_by_size_already_under_target(self, mock_scan_cache_dir):
        """Test size-based cleaning when already under target."""
        mock_cache_info = Mock()
        mock_cache_info.size_on_disk = 1 * 1024**3  # 1GB
        mock_cache_info.size_on_disk_str = "1.0GB"
        mock_cache_info.repos = []

        mock_delete_strategy = Mock()
        mock_delete_strategy.expected_freed_size_str = "0B"
        mock_cache_info.delete_revisions.return_value = mock_delete_strategy

        mock_scan_cache_dir.return_value = mock_cache_info

        result = self.cache_manager.clean_cache_by_size(
            target_size="2GB", strategy="oldest_first", dry_run=True
        )

        assert result == mock_delete_strategy

    @patch("tfbpapi.HfCacheManager.scan_cache_dir")
    def test_clean_unused_revisions(self, mock_scan_cache_dir):
        """Test cleaning unused revisions."""
        # Setup mock with multiple revisions
        mock_cache_info = Mock()

        mock_revision1 = Mock()
        mock_revision1.commit_hash = "abc123"
        mock_revision1.last_modified = (datetime.now() - timedelta(days=1)).timestamp()

        mock_revision2 = Mock()
        mock_revision2.commit_hash = "def456"
        mock_revision2.last_modified = (datetime.now() - timedelta(days=10)).timestamp()

        mock_revision3 = Mock()
        mock_revision3.commit_hash = "ghi789"
        mock_revision3.last_modified = (datetime.now() - timedelta(days=20)).timestamp()

        mock_repo = Mock()
        mock_repo.revisions = [mock_revision1, mock_revision2, mock_revision3]

        mock_cache_info.repos = [mock_repo]
        mock_delete_strategy = Mock()
        mock_delete_strategy.expected_freed_size_str = "1GB"
        mock_cache_info.delete_revisions.return_value = mock_delete_strategy

        mock_scan_cache_dir.return_value = mock_cache_info

        result = self.cache_manager.clean_unused_revisions(keep_latest=2, dry_run=True)

        assert result == mock_delete_strategy
        # Should delete oldest revision (ghi789)
        mock_cache_info.delete_revisions.assert_called_once_with("ghi789")

    @patch("tfbpapi.HfCacheManager.scan_cache_dir")
    def test_auto_clean_cache(self, mock_scan_cache_dir):
        """Test automated cache cleaning."""
        mock_cache_info = Mock()
        mock_cache_info.size_on_disk = 10 * 1024**3  # 10GB
        mock_cache_info.repos = []

        mock_delete_strategy = Mock()
        mock_delete_strategy.expected_freed_size = 1 * 1024**3  # 1GB
        mock_delete_strategy.expected_freed_size_str = "1GB"

        mock_scan_cache_dir.return_value = mock_cache_info

        with patch.object(
            self.cache_manager, "clean_cache_by_age", return_value=mock_delete_strategy
        ):
            with patch.object(
                self.cache_manager,
                "clean_unused_revisions",
                return_value=mock_delete_strategy,
            ):
                with patch.object(
                    self.cache_manager,
                    "clean_cache_by_size",
                    return_value=mock_delete_strategy,
                ):
                    result = self.cache_manager.auto_clean_cache(
                        max_age_days=30,
                        max_total_size="5GB",
                        keep_latest_per_repo=2,
                        dry_run=True,
                    )

                    assert (
                        len(result) == 3
                    )  # All three cleanup strategies should be executed
                    assert all(strategy == mock_delete_strategy for strategy in result)


class TestHfCacheManagerErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("tfbpapi.HfCacheManager.DataCard.__init__", return_value=None):
            self.conn = duckdb.connect(":memory:")
            self.repo_id = "test/repo"
            self.cache_manager = HfCacheManager(self.repo_id, self.conn)

    def test_parse_size_string_invalid_input(self):
        """Test error handling for invalid size strings."""
        with pytest.raises(ValueError):
            self.cache_manager._parse_size_string("invalid")

    @patch("tfbpapi.HfCacheManager.scan_cache_dir")
    def test_clean_cache_invalid_strategy(self, mock_scan_cache_dir):
        """Test error handling for invalid cleanup strategy."""
        mock_cache_info = Mock()
        mock_cache_info.size_on_disk = 5 * 1024**3
        mock_cache_info.repos = []
        mock_scan_cache_dir.return_value = mock_cache_info

        with pytest.raises(ValueError, match="Unknown strategy"):
            self.cache_manager.clean_cache_by_size(
                target_size="1GB",
                strategy="invalid_strategy",  # type: ignore[arg-type]
                dry_run=True,
            )


class TestHfCacheManagerIntegration:
    """Integration tests with real DuckDB operations."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("tfbpapi.HfCacheManager.DataCard.__init__", return_value=None):
            self.conn = duckdb.connect(":memory:")
            self.repo_id = "test/repo"
            self.cache_manager = HfCacheManager(self.repo_id, self.conn)

    def test_metadata_workflow_integration(self, tmpdir):
        """Test complete metadata workflow with real files."""
        # Create temporary parquet file content
        metadata_file = tmpdir.join("metadata.parquet")
        metadata_file.write("dummy_parquet_content")

        # Test the core table creation functionality
        mock_conn = Mock()
        test_cache_manager = HfCacheManager("test/repo", mock_conn)

        # Test _create_duckdb_table_from_files directly
        test_cache_manager._create_duckdb_table_from_files(
            [str(metadata_file)], "metadata_test_metadata"
        )

        # Verify the SQL was generated correctly
        mock_conn.execute.assert_called_once()
        sql_call = mock_conn.execute.call_args[0][0]
        assert "CREATE OR REPLACE VIEW metadata_test_metadata" in sql_call
        assert str(metadata_file) in sql_call

    def test_embedded_metadata_workflow_integration(self):
        """Test complete embedded metadata workflow with real DuckDB operations."""
        # Create real test data in DuckDB
        self.conn.execute(
            """
            CREATE TABLE test_data AS
            SELECT
                'gene_' || (row_number() OVER()) as gene_id,
                CASE
                    WHEN (row_number() OVER()) % 3 = 0 THEN 'treatment_A'
                    WHEN (row_number() OVER()) % 3 = 1 THEN 'treatment_B'
                    ELSE 'control'
                END as experimental_condition,
                random() * 1000 as expression_value
            FROM range(30)
        """
        )

        # Extract embedded metadata
        result = self.cache_manager._extract_embedded_metadata_field(
            "test_data", "experimental_condition", "metadata_test_condition"
        )

        assert result is True

        # Verify the metadata table was created correctly
        metadata_results = self.conn.execute(
            "SELECT value, count FROM metadata_test_condition ORDER BY count DESC"
        ).fetchall()

        assert len(metadata_results) == 3  # Three unique conditions

        # Check that the counts make sense (should be 10 each for 30 total rows)
        total_count = sum(row[1] for row in metadata_results)
        assert total_count == 30

        # Check that conditions are as expected
        conditions = {row[0] for row in metadata_results}
        assert conditions == {"treatment_A", "treatment_B", "control"}

    def test_table_existence_checking_integration(self):
        """Test table existence checking with real DuckDB operations."""
        # Test non-existent table
        assert (
            self.cache_manager._check_metadata_exists_in_duckdb("nonexistent_table")
            is False
        )

        # Create a real table
        self.conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")

        # Test existing table
        assert self.cache_manager._check_metadata_exists_in_duckdb("test_table") is True

        # Test with view
        self.conn.execute("CREATE VIEW test_view AS SELECT * FROM test_table")
        assert self.cache_manager._check_metadata_exists_in_duckdb("test_view") is True


# Fixtures for common test data
@pytest.fixture
def sample_metadata_config():
    """Sample metadata configuration for testing."""
    return Mock(
        config_name="test_metadata",
        description="Test metadata configuration",
        data_files=[Mock(path="metadata.parquet")],
        applies_to=["data_config"],
    )


@pytest.fixture
def sample_data_config():
    """Sample data configuration for testing."""
    return Mock(
        config_name="test_data",
        metadata_fields=["condition", "replicate"],
        dataset_type=DatasetType.ANNOTATED_FEATURES,
    )


@pytest.fixture
def mock_cache_revision():
    """Mock cache revision for testing."""
    revision = Mock()
    revision.commit_hash = "abc123def456"
    revision.last_modified = datetime.now().timestamp()
    revision.size_on_disk = 1024 * 1024 * 100  # 100MB
    return revision


@pytest.fixture
def mock_cache_repo(mock_cache_revision):
    """Mock cache repository for testing."""
    repo = Mock()
    repo.repo_id = "test/repository"
    repo.revisions = [mock_cache_revision]
    repo.size_on_disk = 1024 * 1024 * 100  # 100MB
    repo.size_on_disk_str = "100.0MB"
    return repo


@pytest.fixture
def mock_cache_info(mock_cache_repo):
    """Mock cache info for testing."""
    cache_info = Mock()
    cache_info.cache_dir = "/tmp/cache"
    cache_info.repos = [mock_cache_repo]
    cache_info.size_on_disk = 1024 * 1024 * 100  # 100MB
    cache_info.size_on_disk_str = "100.0MB"

    # Mock delete_revisions method
    def mock_delete_revisions(*revision_hashes):
        strategy = Mock()
        strategy.expected_freed_size = (
            len(revision_hashes) * 1024 * 1024 * 50
        )  # 50MB per revision
        strategy.expected_freed_size_str = f"{len(revision_hashes) * 50}.0MB"
        strategy.delete_content = list(revision_hashes)
        strategy.execute = Mock()
        return strategy

    cache_info.delete_revisions = mock_delete_revisions
    return cache_info
