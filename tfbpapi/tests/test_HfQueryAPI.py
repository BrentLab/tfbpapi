"""Comprehensive tests for HfQueryAPI class."""

import logging
from unittest.mock import MagicMock, Mock, patch

import duckdb
import pandas as pd
import pytest

from tfbpapi.datainfo.models import MetadataRelationship
from tfbpapi.errors import InvalidFilterFieldError
from tfbpapi.HfQueryAPI import HfQueryAPI


class TestHfQueryAPIInit:
    """Test HfQueryAPI initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        conn = duckdb.connect(":memory:")
        repo_id = "test/repo"

        with patch("tfbpapi.HfQueryAPI.HfCacheManager.__init__", return_value=None):
            api = HfQueryAPI(repo_id, duckdb_conn=conn)
            # Manually set properties that would be set by parent
            api.repo_id = repo_id
            api.duckdb_conn = conn
            api._duckdb_conn = conn
            api.logger = logging.getLogger("HfQueryAPI")

            assert api.repo_id == repo_id
            assert api.duckdb_conn == conn
            assert api._duckdb_conn == conn

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        import tempfile

        conn = duckdb.connect(":memory:")
        repo_id = "test/repo"
        token = "test_token"

        with tempfile.TemporaryDirectory() as cache_dir:
            with patch("tfbpapi.HfQueryAPI.HfCacheManager.__init__", return_value=None):
                api = HfQueryAPI(
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                    cache_dir=cache_dir,
                    duckdb_conn=conn,
                )
                # Manually set properties that would be set by parent
                api.repo_id = repo_id
                api.duckdb_conn = conn
                api._duckdb_conn = conn

                assert api.repo_id == repo_id
                assert api.duckdb_conn == conn
                assert api._duckdb_conn == conn
                assert api.repo_type == "model"


class TestHfQueryAPIHelpers:
    """Test HfQueryAPI helper methods."""

    @pytest.fixture
    def mock_api(self):
        """Create a mock HfQueryAPI instance."""
        conn = duckdb.connect(":memory:")
        with patch("tfbpapi.HfQueryAPI.HfCacheManager.__init__", return_value=None):
            api = HfQueryAPI("test/repo", duckdb_conn=conn)
            api.duckdb_conn = conn
            api._duckdb_conn = conn
            api.logger = logging.getLogger("test")
            return api

    def test_get_explicit_metadata(self, mock_api):
        """Test _get_explicit_metadata helper."""
        # Mock config and return value
        mock_config = Mock()
        mock_config.config_name = "test_config"

        # Create mock DataFrame
        expected_df = pd.DataFrame({"field1": [1, 2], "field2": ["a", "b"]})

        # Replace the duckdb_conn with a mock
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchdf.return_value = expected_df
        mock_conn.execute.return_value = mock_result
        mock_api.duckdb_conn = mock_conn

        result = mock_api._get_explicit_metadata(mock_config, "test_table")

        # Verify SQL was executed correctly
        mock_conn.execute.assert_called_once_with("SELECT * FROM test_table")
        pd.testing.assert_frame_equal(result, expected_df)

    def test_get_embedded_metadata(self, mock_api):
        """Test _get_embedded_metadata helper."""
        # Mock config with metadata fields
        mock_config = Mock()
        mock_config.config_name = "test_config"
        mock_config.metadata_fields = ["time", "mechanism"]

        expected_df = pd.DataFrame(
            {"time": [15, 30], "mechanism": ["ZEV", "ZREV"], "count": [100, 50]}
        )

        # Replace the duckdb_conn with a mock
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchdf.return_value = expected_df
        mock_conn.execute.return_value = mock_result
        mock_api.duckdb_conn = mock_conn

        result = mock_api._get_embedded_metadata(mock_config, "test_table")

        # Verify correct SQL was generated
        expected_sql = """
            SELECT DISTINCT time, mechanism, COUNT(*) as count
            FROM test_table
            WHERE time IS NOT NULL AND mechanism IS NOT NULL
            GROUP BY time, mechanism
            ORDER BY count DESC
        """
        mock_conn.execute.assert_called_once()
        actual_sql = mock_conn.execute.call_args[0][0]
        # Normalize whitespace for comparison
        assert " ".join(actual_sql.split()) == " ".join(expected_sql.split())
        pd.testing.assert_frame_equal(result, expected_df)

    def test_get_embedded_metadata_no_fields(self, mock_api):
        """Test _get_embedded_metadata with no metadata fields."""
        mock_config = Mock()
        mock_config.config_name = "test_config"
        mock_config.metadata_fields = None

        with pytest.raises(ValueError, match="has no metadata fields"):
            mock_api._get_embedded_metadata(mock_config, "test_table")

    def test_extract_fields_from_sql_simple(self, mock_api):
        """Test _extract_fields_from_sql with simple SQL."""
        sql = "time = 15 AND mechanism = 'ZEV'"
        fields = mock_api._extract_fields_from_sql(sql)

        assert "time" in fields
        assert "mechanism" in fields
        # Note: The current regex may pick up quoted strings as identifiers
        # This is a limitation we accept for simplicity
        assert "15" not in fields  # Should not include numeric literals

    def test_extract_fields_from_sql_complex(self, mock_api):
        """Test _extract_fields_from_sql with complex SQL."""
        sql = "field1 IN (1, 2, 3) AND field2 IS NOT NULL AND field3 LIKE '%test%'"
        fields = mock_api._extract_fields_from_sql(sql)

        assert "field1" in fields
        assert "field2" in fields
        assert "field3" in fields
        assert "NULL" not in fields  # SQL keyword should be excluded
        assert "LIKE" not in fields  # SQL keyword should be excluded

    def test_extract_fields_from_sql_quoted(self, mock_api):
        """Test _extract_fields_from_sql with quoted identifiers."""
        sql = "\"quoted_field\" = 1 AND 'another_field' > 5"
        fields = mock_api._extract_fields_from_sql(sql)

        assert "quoted_field" in fields
        assert "another_field" in fields

    def test_validate_metadata_fields_success(self, mock_api):
        """Test _validate_metadata_fields with valid fields."""
        # Mock get_metadata to return DataFrame with expected columns
        metadata_df = pd.DataFrame(
            {"time": [15, 30], "mechanism": ["ZEV", "ZREV"], "restriction": ["P", "A"]}
        )

        with patch.object(mock_api, "get_metadata", return_value=metadata_df):
            # Should not raise any exception
            mock_api._validate_metadata_fields("test_config", ["time", "mechanism"])

    def test_validate_metadata_fields_invalid(self, mock_api):
        """Test _validate_metadata_fields with invalid fields."""
        metadata_df = pd.DataFrame({"time": [15, 30], "mechanism": ["ZEV", "ZREV"]})

        with patch.object(mock_api, "get_metadata", return_value=metadata_df):
            with pytest.raises(InvalidFilterFieldError) as exc_info:
                mock_api._validate_metadata_fields(
                    "test_config", ["invalid_field", "time"]
                )

            error = exc_info.value
            assert "invalid_field" in error.invalid_fields
            assert "time" not in error.invalid_fields
            assert "time" in error.available_fields
            assert error.config_name == "test_config"

    def test_validate_metadata_fields_empty_metadata(self, mock_api):
        """Test _validate_metadata_fields with empty metadata."""
        empty_df = pd.DataFrame()

        with patch.object(mock_api, "get_metadata", return_value=empty_df):
            with pytest.raises(InvalidFilterFieldError) as exc_info:
                mock_api._validate_metadata_fields("test_config", ["any_field"])

            error = exc_info.value
            assert error.invalid_fields == ["any_field"]
            assert error.available_fields == []

    def test_validate_metadata_fields_empty_list(self, mock_api):
        """Test _validate_metadata_fields with empty field list."""
        # Should not call get_metadata or raise any exception
        with patch.object(mock_api, "get_metadata") as mock_get_metadata:
            mock_api._validate_metadata_fields("test_config", [])
            mock_get_metadata.assert_not_called()


class TestHfQueryAPIMainMethods:
    """Test HfQueryAPI main methods."""

    @pytest.fixture
    def mock_api(self):
        """Create a mock HfQueryAPI instance."""
        conn = duckdb.connect(":memory:")
        with patch("tfbpapi.HfQueryAPI.HfCacheManager.__init__", return_value=None):
            api = HfQueryAPI("test/repo", duckdb_conn=conn)
            # Set up all necessary attributes
            api.repo_id = "test/repo"
            api.duckdb_conn = conn
            api._duckdb_conn = conn
            api.logger = logging.getLogger("test")
            api._table_filters = {}

            # Set up the internal dataset card attribute
            api._dataset_card = Mock()
            return api

    def test_get_metadata_explicit_config(self, mock_api):
        """Test get_metadata with explicit metadata config."""
        # Setup mock configurations
        explicit_config = Mock()
        explicit_config.config_name = "metadata_config"
        explicit_config.applies_to = None  # This config doesn't apply to others

        # Mock the metadata relationship
        relationship = MetadataRelationship(
            data_config="some_data",
            metadata_config="metadata_config",
            relationship_type="explicit",
        )

        # Mock the config loading and table setup
        mock_config_result = {"success": True, "table_name": "test_metadata_table"}

        expected_df = pd.DataFrame(
            {"sample_id": ["sample1", "sample2"], "condition": ["ctrl", "treatment"]}
        )

        with (
            patch.object(
                mock_api, "get_metadata_relationships", return_value=[relationship]
            ),
            patch.object(mock_api, "get_config", return_value=explicit_config),
            patch.object(
                mock_api, "_get_metadata_for_config", return_value=mock_config_result
            ),
            patch.object(mock_api, "_get_explicit_metadata", return_value=expected_df),
        ):

            result = mock_api.get_metadata("metadata_config")

            mock_api._get_explicit_metadata.assert_called_once_with(
                explicit_config, "test_metadata_table"
            )
            pd.testing.assert_frame_equal(result, expected_df)

    def test_get_metadata_embedded_config(self, mock_api):
        """Test get_metadata with embedded metadata config."""
        # Setup mock configurations
        embedded_config = Mock()
        embedded_config.config_name = "data_config"
        embedded_config.metadata_fields = ["time", "mechanism"]

        # Mock the metadata relationship
        relationship = MetadataRelationship(
            data_config="data_config",
            metadata_config="data_config_embedded",
            relationship_type="embedded",
        )
        mock_config_result = {"success": True, "table_name": "test_data_table"}

        expected_df = pd.DataFrame(
            {"time": [15, 30], "mechanism": ["ZEV", "ZREV"], "count": [100, 50]}
        )

        with (
            patch.object(
                mock_api, "get_metadata_relationships", return_value=[relationship]
            ),
            patch.object(mock_api, "get_config", return_value=embedded_config),
            patch.object(
                mock_api, "_get_metadata_for_config", return_value=mock_config_result
            ),
            patch.object(mock_api, "_get_embedded_metadata", return_value=expected_df),
        ):

            result = mock_api.get_metadata("data_config")

            mock_api._get_embedded_metadata.assert_called_once_with(
                embedded_config, "test_data_table"
            )
            pd.testing.assert_frame_equal(result, expected_df)

    def test_get_metadata_applied_config(self, mock_api):
        """Test get_metadata with config that has metadata applied to it."""
        # Setup a metadata config that applies to another config
        metadata_config = Mock()
        metadata_config.config_name = "experiment_metadata"
        metadata_config.applies_to = ["data_config", "other_data_config"]

        # Mock the metadata relationship
        relationship = MetadataRelationship(
            data_config="data_config",
            metadata_config="experiment_metadata",
            relationship_type="explicit",
        )
        mock_config_result = {"success": True, "table_name": "test_metadata_table"}

        expected_df = pd.DataFrame(
            {"experiment_id": ["exp1", "exp2"], "condition": ["ctrl", "treatment"]}
        )

        with (
            patch.object(
                mock_api, "get_metadata_relationships", return_value=[relationship]
            ),
            patch.object(mock_api, "get_config", return_value=metadata_config),
            patch.object(
                mock_api, "_get_metadata_for_config", return_value=mock_config_result
            ),
            patch.object(mock_api, "_get_explicit_metadata", return_value=expected_df),
        ):

            # Request metadata for a config that appears in applies_to
            result = mock_api.get_metadata("data_config")

            # Should return the metadata from the config that applies to it
            mock_api._get_explicit_metadata.assert_called_once_with(
                metadata_config, "test_metadata_table"
            )
            pd.testing.assert_frame_equal(result, expected_df)

    def test_get_metadata_config_not_found(self, mock_api):
        """Test get_metadata with non-existent config when other configs exist."""
        # Setup a relationship for a different config
        relationship = MetadataRelationship(
            data_config="other_data",
            metadata_config="other_config",
            relationship_type="explicit",
        )
        with patch.object(
            mock_api, "get_metadata_relationships", return_value=[relationship]
        ):
            with pytest.raises(ValueError, match="Config 'nonexistent' not found"):
                mock_api.get_metadata("nonexistent")

    def test_get_metadata_no_metadata_sources(self, mock_api):
        """Test get_metadata when no metadata sources are available."""
        with patch.object(mock_api, "get_metadata_relationships", return_value=[]):
            result = mock_api.get_metadata("any_config")
            assert result.empty

    def test_get_metadata_load_failure(self, mock_api):
        """Test get_metadata when config loading fails."""
        config = Mock()
        config.config_name = "test_config"
        config.applies_to = None

        # Mock the metadata relationship
        relationship = MetadataRelationship(
            data_config="some_data",
            metadata_config="test_config",
            relationship_type="explicit",
        )
        mock_config_result = {"success": False}

        with (
            patch.object(
                mock_api, "get_metadata_relationships", return_value=[relationship]
            ),
            patch.object(mock_api, "get_config", return_value=config),
            patch.object(
                mock_api, "_get_metadata_for_config", return_value=mock_config_result
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to load data for config"):
                mock_api.get_metadata("test_config")

    def test_set_filter_valid_fields(self, mock_api):
        """Test set_filter with valid field names."""
        with patch.object(mock_api, "_validate_metadata_fields") as mock_validate:
            mock_api.set_filter("test_config", time=15, mechanism="ZEV")

            mock_validate.assert_called_once_with("test_config", ["time", "mechanism"])
            assert (
                mock_api._table_filters["test_config"]
                == "time = 15 AND mechanism = 'ZEV'"
            )

    def test_set_filter_clear_on_empty(self, mock_api):
        """Test set_filter clears filter when no kwargs provided."""
        # Set an initial filter
        mock_api._table_filters["test_config"] = "existing_filter"

        with patch.object(mock_api, "clear_filter") as mock_clear:
            mock_api.set_filter("test_config")
            mock_clear.assert_called_once_with("test_config")

    def test_set_filter_various_types(self, mock_api):
        """Test set_filter with different value types."""
        with patch.object(mock_api, "_validate_metadata_fields"):
            mock_api.set_filter(
                "test_config",
                string_field="text",
                numeric_field=42,
                null_field=None,
                bool_field=True,
            )

            expected = (
                "string_field = 'text' AND numeric_field = 42 AND "
                "null_field IS NULL AND bool_field = True"
            )
            assert mock_api._table_filters["test_config"] == expected

    def test_set_sql_filter_with_validation(self, mock_api):
        """Test set_sql_filter with field validation enabled."""
        sql_where = "time IN (15, 30) AND mechanism = 'ZEV'"

        with (
            patch.object(
                mock_api, "_extract_fields_from_sql", return_value=["time", "mechanism"]
            ) as mock_extract,
            patch.object(mock_api, "_validate_metadata_fields") as mock_validate,
        ):

            mock_api.set_sql_filter("test_config", sql_where)

            mock_extract.assert_called_once_with(sql_where)
            mock_validate.assert_called_once_with("test_config", ["time", "mechanism"])
            assert mock_api._table_filters["test_config"] == sql_where

    def test_set_sql_filter_without_validation(self, mock_api):
        """Test set_sql_filter with field validation disabled."""
        sql_where = "complex_function(field1, field2) > 0"

        with (
            patch.object(mock_api, "_extract_fields_from_sql") as mock_extract,
            patch.object(mock_api, "_validate_metadata_fields") as mock_validate,
        ):

            mock_api.set_sql_filter("test_config", sql_where, validate_fields=False)

            mock_extract.assert_not_called()
            mock_validate.assert_not_called()
            assert mock_api._table_filters["test_config"] == sql_where

    def test_set_sql_filter_clear_on_empty(self, mock_api):
        """Test set_sql_filter clears filter when empty SQL provided."""
        mock_api._table_filters["test_config"] = "existing_filter"

        with patch.object(mock_api, "clear_filter") as mock_clear:
            mock_api.set_sql_filter("test_config", "")
            mock_clear.assert_called_once_with("test_config")

    def test_clear_filter(self, mock_api):
        """Test clear_filter removes stored filter."""
        mock_api._table_filters["test_config"] = "some_filter"

        mock_api.clear_filter("test_config")

        assert "test_config" not in mock_api._table_filters

    def test_clear_filter_nonexistent(self, mock_api):
        """Test clear_filter with non-existent config."""
        # Should not raise an error
        mock_api.clear_filter("nonexistent_config")

    def test_get_current_filter_exists(self, mock_api):
        """Test get_current_filter returns existing filter."""
        expected_filter = "time = 15"
        mock_api._table_filters["test_config"] = expected_filter

        result = mock_api.get_current_filter("test_config")
        assert result == expected_filter

    def test_get_current_filter_not_exists(self, mock_api):
        """Test get_current_filter returns None for non-existent filter."""
        result = mock_api.get_current_filter("nonexistent_config")
        assert result is None


class TestHfQueryAPIErrorHandling:
    """Test HfQueryAPI error handling and edge cases."""

    @pytest.fixture
    def mock_api(self):
        """Create a mock HfQueryAPI instance."""
        conn = duckdb.connect(":memory:")
        with patch("tfbpapi.HfQueryAPI.HfCacheManager.__init__", return_value=None):
            api = HfQueryAPI("test/repo", duckdb_conn=conn)
            # Set up all necessary attributes
            api.repo_id = "test/repo"
            api.duckdb_conn = conn
            api._duckdb_conn = conn
            api.logger = logging.getLogger("test")
            api._table_filters = {}

            # Set up the internal dataset card attribute
            api._dataset_card = Mock()
            return api

    def test_set_filter_validation_error_propagates(self, mock_api):
        """Test that InvalidFilterFieldError from validation propagates."""
        error = InvalidFilterFieldError(
            config_name="test_config",
            invalid_fields=["invalid_field"],
            available_fields=["valid_field"],
        )

        with patch.object(mock_api, "_validate_metadata_fields", side_effect=error):
            with pytest.raises(InvalidFilterFieldError) as exc_info:
                mock_api.set_filter("test_config", invalid_field="value")

            assert exc_info.value.config_name == "test_config"
            assert "invalid_field" in exc_info.value.invalid_fields

    def test_set_sql_filter_validation_error_propagates(self, mock_api):
        """Test that InvalidFilterFieldError from SQL validation propagates."""
        error = InvalidFilterFieldError(
            config_name="test_config",
            invalid_fields=["nonexistent"],
            available_fields=["time", "mechanism"],
        )

        with (
            patch.object(
                mock_api, "_extract_fields_from_sql", return_value=["nonexistent"]
            ),
            patch.object(mock_api, "_validate_metadata_fields", side_effect=error),
        ):

            with pytest.raises(InvalidFilterFieldError) as exc_info:
                mock_api.set_sql_filter("test_config", "nonexistent = 1")

            assert exc_info.value.config_name == "test_config"

    def test_get_metadata_query_error_propagates(self, mock_api):
        """Test that query errors in get_metadata propagate."""
        config = Mock()
        config.config_name = "test_config"
        config.applies_to = None

        # Mock the metadata relationship
        relationship = MetadataRelationship(
            data_config="some_data",
            metadata_config="test_config",
            relationship_type="explicit",
        )

        mock_config_result = {"success": True, "table_name": "test_table"}

        with (
            patch.object(
                mock_api, "get_metadata_relationships", return_value=[relationship]
            ),
            patch.object(mock_api, "get_config", return_value=config),
            patch.object(
                mock_api, "_get_metadata_for_config", return_value=mock_config_result
            ),
            patch.object(
                mock_api,
                "_get_explicit_metadata",
                side_effect=Exception("Query failed"),
            ),
        ):

            with pytest.raises(Exception, match="Query failed"):
                mock_api.get_metadata("test_config")

    def test_validate_metadata_fields_get_metadata_error_logged(self, mock_api):
        """Test that non-InvalidFilterFieldError exceptions in validation are logged."""
        with (
            patch.object(
                mock_api, "get_metadata", side_effect=Exception("Network error")
            ),
            patch.object(mock_api.logger, "warning") as mock_warning,
        ):

            # Should not raise, but should log warning
            mock_api._validate_metadata_fields("test_config", ["field1"])

            mock_warning.assert_called_once()
            assert "Could not validate filter fields" in mock_warning.call_args[0][0]

    def test_extract_fields_edge_cases(self, mock_api):
        """Test _extract_fields_from_sql with various edge cases."""
        # Empty string
        assert mock_api._extract_fields_from_sql("") == []

        # Only SQL keywords
        fields = mock_api._extract_fields_from_sql("AND OR NOT NULL")
        assert len(fields) == 0

        # Mixed quotes and operators
        fields = mock_api._extract_fields_from_sql(
            '"field1" >= "field2" AND field3 <= field4'
        )
        assert "field1" in fields
        assert "field2" in fields
        assert "field3" in fields
        assert "field4" in fields

        # Function calls should be excluded
        fields = mock_api._extract_fields_from_sql(
            "UPPER(field1) = 'VALUE' AND field2 > MAX(field3)"
        )
        assert "field1" in fields
        assert "field2" in fields
        assert "field3" in fields
        assert "UPPER" not in fields
        assert "MAX" not in fields

        # Numeric literals should be excluded
        fields = mock_api._extract_fields_from_sql("field1 = 123.45 AND field2 = -67")
        assert "field1" in fields
        assert "field2" in fields
        assert "123" not in fields
        assert "45" not in fields
        assert "67" not in fields

    def test_extract_fields_complex_sql(self, mock_api):
        """Test _extract_fields_from_sql with complex SQL patterns."""
        complex_sql = """
        (field1 IN (1, 2, 3) OR field2 IS NOT NULL)
        AND field3 LIKE '%pattern%'
        AND "quoted field" BETWEEN 'start' AND 'end'
        AND field4 > COALESCE(field5, 0)
        """

        fields = mock_api._extract_fields_from_sql(complex_sql)

        expected_fields = [
            "field1",
            "field2",
            "field3",
            "quoted field",
            "field4",
            "field5",
        ]
        for field in expected_fields:
            assert field in fields, f"Field '{field}' should be extracted from SQL"

        # These should not be extracted (SQL keywords and function names)
        unwanted = ["COALESCE", "LIKE", "BETWEEN", "NULL"]
        for unwanted_item in unwanted:
            assert (
                unwanted_item not in fields
            ), f"'{unwanted_item}' should not be extracted"

        # String literals should not be extracted
        string_literals = ["start", "end", "pattern"]
        for literal in string_literals:
            assert (
                literal not in fields
            ), f"String literal '{literal}' should not be extracted"

    def test_extract_fields_in_clause_with_quoted_values(self, mock_api):
        """Test _extract_fields_from_sql with IN clause containing quoted values."""
        # This is the exact pattern from the user's error case
        gene_ids = ["YNL199C", "YDL106C", "YLR098C", "YNR009W", "YLR176C"]
        regulator_clause = "(" + ", ".join(f"'{gene_id}'" for gene_id in gene_ids) + ")"

        sql = f"""
            time = 15
            AND mechanism = 'ZEV'
            AND restriction = 'P'
            AND regulator_locus_tag IN {regulator_clause}
        """

        fields = mock_api._extract_fields_from_sql(sql)

        # Should extract field names
        expected_fields = ["time", "mechanism", "restriction", "regulator_locus_tag"]
        for field in expected_fields:
            assert field in fields, f"Field '{field}' should be extracted from SQL"

        # Should NOT extract string literals or gene IDs
        unwanted_values = ["ZEV", "P"] + gene_ids
        for value in unwanted_values:
            assert (
                value not in fields
            ), f"String literal '{value}' should not be extracted as field"

        # Should NOT extract numeric literals
        assert "15" not in fields

    def test_extract_fields_various_comparison_operators(self, mock_api):
        """Test _extract_fields_from_sql with various comparison operators and string
        values."""
        sql = """
        field1 = 'value1' AND field2 != 'value2'
        AND field3 <> 'value3' AND field4 > 'value4'
        AND field5 <= 'value5' AND field6 >= 'value6'
        AND field7 LIKE 'pattern%' AND field8 NOT LIKE '%other%'
        """

        fields = mock_api._extract_fields_from_sql(sql)

        # Should extract field names
        expected_fields = [
            "field1",
            "field2",
            "field3",
            "field4",
            "field5",
            "field6",
            "field7",
            "field8",
        ]
        for field in expected_fields:
            assert field in fields, f"Field '{field}' should be extracted from SQL"

        # Should NOT extract string values
        unwanted_values = [
            "value1",
            "value2",
            "value3",
            "value4",
            "value5",
            "value6",
            "pattern",
            "other",
        ]
        for value in unwanted_values:
            assert (
                value not in fields
            ), f"String literal '{value}' should not be extracted as field"

    def test_extract_fields_between_clause(self, mock_api):
        """Test _extract_fields_from_sql with BETWEEN clause containing string
        values."""
        sql = (
            "field1 BETWEEN 'start_value' AND 'end_value' AND field2 BETWEEN 10 AND 20"
        )

        fields = mock_api._extract_fields_from_sql(sql)

        # Should extract field names
        assert "field1" in fields
        assert "field2" in fields

        # Should NOT extract BETWEEN values
        assert "start_value" not in fields
        assert "end_value" not in fields
        assert "10" not in fields
        assert "20" not in fields

    def test_get_metadata_table_name_missing(self, mock_api):
        """Test get_metadata when table_name is missing from config result."""
        config = Mock()
        config.config_name = "test_config"
        config.applies_to = None

        # Mock the metadata relationship
        relationship = MetadataRelationship(
            data_config="some_data",
            metadata_config="test_config",
            relationship_type="explicit",
        )
        # Success but no table name
        mock_config_result = {"success": True, "table_name": None}

        with (
            patch.object(
                mock_api, "get_metadata_relationships", return_value=[relationship]
            ),
            patch.object(mock_api, "get_config", return_value=config),
            patch.object(
                mock_api, "_get_metadata_for_config", return_value=mock_config_result
            ),
        ):
            with pytest.raises(RuntimeError, match="No table name for config"):
                mock_api.get_metadata("test_config")

    def test_filter_methods_whitespace_handling(self, mock_api):
        """Test that filter methods handle whitespace correctly."""
        # set_sql_filter should strip whitespace
        with (
            patch.object(mock_api, "_extract_fields_from_sql", return_value=[]),
            patch.object(mock_api, "_validate_metadata_fields"),
        ):

            mock_api.set_sql_filter("test_config", "  field = 1  ")
            assert mock_api._table_filters["test_config"] == "field = 1"

        # Empty/whitespace-only SQL should clear filter
        with patch.object(mock_api, "clear_filter") as mock_clear:
            mock_api.set_sql_filter("test_config", "   ")
            mock_clear.assert_called_once_with("test_config")


class TestInvalidFilterFieldError:
    """Test the InvalidFilterFieldError exception."""

    def test_error_message_formatting(self):
        """Test that error message is formatted correctly."""
        error = InvalidFilterFieldError(
            config_name="test_config",
            invalid_fields=["field1", "field2"],
            available_fields=["valid1", "valid2", "valid3"],
        )

        message = str(error)
        assert "test_config" in message
        assert "'field1'" in message
        assert "'field2'" in message
        assert "Available fields:" in message
        assert "valid1" in message

    def test_error_with_no_available_fields(self):
        """Test error message when no fields are available."""
        error = InvalidFilterFieldError(
            config_name="empty_config",
            invalid_fields=["any_field"],
            available_fields=[],
        )

        message = str(error)
        assert "No fields available" in message

    def test_error_attributes(self):
        """Test that error attributes are set correctly."""
        invalid_fields = ["bad1", "bad2"]
        available_fields = ["good1", "good2"]

        error = InvalidFilterFieldError(
            config_name="test_config",
            invalid_fields=invalid_fields,
            available_fields=available_fields,
        )

        assert error.config_name == "test_config"
        assert error.invalid_fields == invalid_fields
        assert error.available_fields == sorted(available_fields)
        assert error.details["config_name"] == "test_config"
        assert error.details["invalid_fields"] == invalid_fields
        assert error.details["available_fields"] == sorted(available_fields)
