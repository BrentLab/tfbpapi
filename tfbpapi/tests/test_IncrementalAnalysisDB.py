import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tfbpapi.IncrementalAnalysisDB import IncrementalAnalysisDB


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        yield str(db_path)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [10.5, 20.3, 15.7]}
    )


@pytest.fixture
def analysis_db(temp_db_path):
    """Create IncrementalAnalysisDB instance for testing."""
    return IncrementalAnalysisDB(temp_db_path)


class TestIncrementalAnalysisDB:

    def test_init_creates_database_and_metadata_table(self, temp_db_path):
        """Test that initialization creates the database file and metadata table."""
        db = IncrementalAnalysisDB(temp_db_path)

        # Check database file exists
        assert Path(temp_db_path).exists()

        # Check metadata table exists
        result = db.conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_name='analysis_metadata' AND table_schema='main'
        """
        ).fetchall()
        assert len(result) == 1

        db.conn.close()

    def test_init_creates_parent_directories(self):
        """Test that initialization creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "path" / "test.db"
            db = IncrementalAnalysisDB(str(nested_path))

            assert nested_path.parent.exists()
            assert nested_path.exists()
            db.conn.close()

    def test_append_results_new_table(self, analysis_db, sample_dataframe):
        """Test appending results to a new table."""
        records_added = analysis_db.append_results(
            new_results=sample_dataframe,
            table_name="test_table",
            analysis_type="test_analysis",
            parameters={"param1": "value1"},
            description="Test description",
        )

        assert records_added == 3

        # Check data was inserted
        result = analysis_db.conn.execute("SELECT * FROM test_table").fetchdf()
        pd.testing.assert_frame_equal(result, sample_dataframe)

        # Check metadata was inserted
        metadata = analysis_db.conn.execute(
            """
            SELECT * FROM analysis_metadata WHERE table_name = 'test_table'
        """
        ).fetchdf()

        assert len(metadata) == 1
        assert metadata.iloc[0]["analysis_type"] == "test_analysis"
        assert metadata.iloc[0]["total_records"] == 3
        assert json.loads(metadata.iloc[0]["parameters"]) == {"param1": "value1"}
        assert metadata.iloc[0]["description"] == "Test description"

    def test_append_results_existing_table(self, analysis_db, sample_dataframe):
        """Test appending results to an existing table."""
        # First append
        analysis_db.append_results(sample_dataframe, "test_table")

        # Second append with new data
        new_data = pd.DataFrame(
            {"id": [4, 5], "name": ["D", "E"], "value": [25.1, 30.9]}
        )

        records_added = analysis_db.append_results(new_data, "test_table")
        assert records_added == 2

        # Check total records
        result = analysis_db.conn.execute(
            "SELECT COUNT(*) as count FROM test_table"
        ).fetchdf()
        assert result.iloc[0]["count"] == 5

        # Check metadata updated
        metadata = analysis_db.conn.execute(
            """
            SELECT total_records FROM analysis_metadata WHERE table_name = 'test_table'
        """
        ).fetchdf()
        assert metadata.iloc[0]["total_records"] == 5

    def test_append_results_with_deduplication(self, analysis_db):
        """Test appending results with deduplication."""
        initial_data = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [10.5, 20.3, 15.7]}
        )

        analysis_db.append_results(initial_data, "test_table", deduplicate_on=["id"])

        # Append data with some duplicates
        new_data = pd.DataFrame(
            {
                "id": [2, 3, 4],  # 2 and 3 are duplicates
                "name": ["B2", "C2", "D"],
                "value": [20.3, 15.7, 25.1],
            }
        )

        records_added = analysis_db.append_results(
            new_data, "test_table", deduplicate_on=["id"]
        )

        # Only record with id=4 should be added
        assert records_added == 1

        # Check total records
        result = analysis_db.conn.execute(
            "SELECT COUNT(*) as count FROM test_table"
        ).fetchdf()
        assert result.iloc[0]["count"] == 4

    def test_update_results(self, analysis_db, sample_dataframe):
        """Test updating existing results."""
        # First insert data
        analysis_db.append_results(sample_dataframe, "test_table")

        # Update data
        updated_data = pd.DataFrame(
            {"id": [1, 2], "name": ["A_updated", "B_updated"], "value": [100.5, 200.3]}
        )

        records_updated = analysis_db.update_results(
            updated_data, "test_table", key_columns=["id"]
        )

        assert records_updated == 2

        # Check data was updated
        result = analysis_db.conn.execute(
            """
            SELECT * FROM test_table WHERE id IN (1, 2) ORDER BY id
        """
        ).fetchdf()

        assert result.iloc[0]["name"] == "A_updated"
        assert result.iloc[1]["name"] == "B_updated"

    def test_get_results(self, analysis_db, sample_dataframe):
        """Test retrieving results."""
        analysis_db.append_results(sample_dataframe, "test_table")

        # Get all results
        result = analysis_db.get_results("test_table")
        pd.testing.assert_frame_equal(result, sample_dataframe)

        # Get results with filter
        filtered_result = analysis_db.get_results("test_table", filters={"id": [1, 2]})

        expected = sample_dataframe[sample_dataframe["id"].isin([1, 2])]
        pd.testing.assert_frame_equal(filtered_result.reset_index(drop=True), expected)

    def test_get_results_with_limit(self, analysis_db, sample_dataframe):
        """Test retrieving results with limit."""
        analysis_db.append_results(sample_dataframe, "test_table")

        result = analysis_db.get_results("test_table", limit=2)
        assert len(result) == 2

    def test_query_method(self, analysis_db, sample_dataframe):
        """Test direct SQL query execution."""
        analysis_db.append_results(sample_dataframe, "test_table")

        # Test basic query
        result = analysis_db.query("SELECT * FROM test_table")
        assert len(result) == len(sample_dataframe)
        pd.testing.assert_frame_equal(result, sample_dataframe)

        # Test query with WHERE clause
        result = analysis_db.query("SELECT * FROM test_table WHERE id = 1")
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1

        # Test query with aggregation
        result = analysis_db.query("SELECT COUNT(*) as count FROM test_table")
        assert result.iloc[0]["count"] == len(sample_dataframe)

        # Test query with complex SQL
        result = analysis_db.query(
            """
            SELECT name, AVG(value) as avg_value
            FROM test_table
            GROUP BY name
            ORDER BY name
        """
        )
        assert len(result) == 3  # Should have 3 distinct names
        assert "avg_value" in result.columns

    def test_table_exists(self, analysis_db, sample_dataframe):
        """Test checking if table exists."""
        assert not analysis_db.table_exists("test_table")

        analysis_db.append_results(sample_dataframe, "test_table")
        assert analysis_db.table_exists("test_table")

    def test_drop_table(self, analysis_db, sample_dataframe):
        """Test dropping a table."""
        analysis_db.append_results(sample_dataframe, "test_table")
        assert analysis_db.table_exists("test_table")

        analysis_db.drop_table("test_table")
        assert not analysis_db.table_exists("test_table")

        # Check metadata was also removed
        metadata = analysis_db.conn.execute(
            """
            SELECT * FROM analysis_metadata WHERE table_name = 'test_table'
        """
        ).fetchdf()
        assert len(metadata) == 0

    def test_get_table_info(self, analysis_db, sample_dataframe):
        """Test getting table information."""
        analysis_db.append_results(
            sample_dataframe,
            "test_table",
            analysis_type="test_analysis",
            parameters={"param1": "value1"},
            description="Test description",
        )

        info = analysis_db.get_table_info("test_table")

        assert info["table_name"] == "test_table"
        assert info["total_records"] == 3
        assert info["analysis_type"] == "test_analysis"
        assert json.loads(info["parameters"]) == {"param1": "value1"}
        assert info["description"] == "Test description"

    def test_list_tables(self, analysis_db, sample_dataframe):
        """Test listing all tables."""
        # Initially should be empty (except metadata table)
        tables = analysis_db.list_tables()
        assert "analysis_metadata" in tables

        # Add some tables
        analysis_db.append_results(sample_dataframe, "table1")
        analysis_db.append_results(sample_dataframe, "table2")

        tables = analysis_db.list_tables()
        assert "table1" in tables
        assert "table2" in tables
        assert "analysis_metadata" in tables

    def test_get_table_schema(self, analysis_db, sample_dataframe):
        """Test getting table schema."""
        analysis_db.append_results(sample_dataframe, "test_table")

        schema = analysis_db.get_table_schema("test_table")

        # Should have columns from sample dataframe
        column_names = [col["column_name"] for col in schema]
        assert "id" in column_names
        assert "name" in column_names
        assert "value" in column_names

    def test_close_connection(self, analysis_db):
        """Test closing database connection."""
        analysis_db.close()

        # Connection should be closed
        with pytest.raises(Exception):
            analysis_db.conn.execute("SELECT 1")

    def test_context_manager(self, temp_db_path, sample_dataframe):
        """Test using IncrementalAnalysisDB as context manager."""
        with IncrementalAnalysisDB(temp_db_path) as db:
            db.append_results(sample_dataframe, "test_table")
            assert db.table_exists("test_table")

        # Connection should be closed after context exit
        with pytest.raises(Exception):
            db.conn.execute("SELECT 1")

    @patch("tfbpapi.IncrementalAnalysisDB.logging.getLogger")
    def test_logging_setup(self, mock_get_logger, temp_db_path):
        """Test that logging is properly configured."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        db = IncrementalAnalysisDB(temp_db_path)

        mock_get_logger.assert_called_once_with("tfbpapi.IncrementalAnalysisDB")
        assert db.logger == mock_logger
        db.conn.close()

    def test_error_handling_nonexistent_table(self, analysis_db):
        """Test error handling for operations on nonexistent tables."""
        with pytest.raises(Exception):
            analysis_db.get_results("nonexistent_table")

        with pytest.raises(Exception):
            analysis_db.get_table_info("nonexistent_table")

    def test_empty_dataframe_append(self, analysis_db):
        """Test appending empty DataFrame."""
        empty_df = pd.DataFrame()

        records_added = analysis_db.append_results(empty_df, "empty_table")
        assert records_added == 0

        # Table should not be created for empty DataFrame
        assert not analysis_db.table_exists("empty_table")
