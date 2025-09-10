import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tfbpapi.HfRankResponse import HfRankResponse
from tfbpapi.IncrementalAnalysisDB import IncrementalAnalysisDB


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        yield str(db_path)


@pytest.fixture
def analysis_db(temp_db_path):
    """Create IncrementalAnalysisDB instance for testing."""
    return IncrementalAnalysisDB(temp_db_path)


@pytest.fixture
def rank_response(analysis_db):
    """Create HfRankResponse instance for testing."""
    return HfRankResponse(analysis_db)


@pytest.fixture
def mock_ranking_api():
    """Create mock ranking API."""
    api = MagicMock()
    api.get_table_filter.return_value = None
    api.query.return_value = pd.DataFrame(
        {
            "regulator_locus_tag": ["REG1", "REG2", "REG3"],
        }
    )
    api._duckdb_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        {
            "regulator_locus_tag": ["REG1", "REG1", "REG2", "REG2", "REG3"],
            "target_locus_tag": ["TGT1", "TGT2", "TGT1", "TGT3", "TGT1"],
            "binding_score": [10.5, 8.2, 9.1, 7.8, 6.5],
        }
    )
    return api


@pytest.fixture
def mock_response_api():
    """Create mock response API."""
    api = MagicMock()
    api.get_table_filter.return_value = None
    api._duckdb_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        {
            "regulator_locus_tag": ["REG1", "REG1", "REG2"],
            "target_locus_tag": ["TGT1", "TGT2", "TGT1"],
            "log2fc": [2.1, -1.5, 1.8],
        }
    )
    return api


@pytest.fixture
def sample_computed_data():
    """Create sample computed rank response data."""
    return pd.DataFrame(
        {
            "regulator_locus_tag": ["REG1", "REG1", "REG1", "REG2", "REG2", "REG3"],
            "target_locus_tag": ["TGT1", "TGT2", "TGT3", "TGT1", "TGT2", "TGT1"],
            "binding_score": [10.5, 8.2, 7.1, 9.1, 7.8, 6.5],
            "log2fc": [2.1, -1.5, None, 1.8, None, None],
            "responsive": [1, 1, 0, 1, 0, 0],
            "bin_label": [5, 5, 10, 5, 10, 5],
            "cumulative_responsive": [2, 2, 2, 1, 1, 0],
        }
    )


class TestHfRankResponse:

    def test_init(self, analysis_db):
        """Test HfRankResponse initialization."""
        rr = HfRankResponse(analysis_db)
        assert rr.db == analysis_db
        assert rr.logger is not None

    def test_get_comparisons_empty(self, rank_response):
        """Test getting comparisons when none exist."""
        comparisons = rank_response.get_comparisons()
        assert comparisons == []

    def test_get_comparisons_with_data(self, rank_response, sample_computed_data):
        """Test getting comparisons with existing data."""
        # Add some test data
        rank_response.db.append_results(
            sample_computed_data, "rank_response_test_comparison_1"
        )
        rank_response.db.append_results(
            sample_computed_data, "rank_response_test_comparison_2"
        )

        comparisons = rank_response.get_comparisons()
        assert "test_comparison_1" in comparisons
        assert "test_comparison_2" in comparisons
        assert len(comparisons) == 2

    @patch("tfbpapi.HfRankResponse.duckdb.connect")
    def test_compute_new_comparison(
        self,
        mock_duckdb_connect,
        rank_response,
        mock_ranking_api,
        mock_response_api,
        sample_computed_data,
    ):
        """Test computing a new comparison."""
        # Mock the temporary connection
        mock_temp_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_temp_conn
        mock_temp_conn.execute.return_value.fetchdf.return_value = sample_computed_data

        result = rank_response.compute(
            ranking_api=mock_ranking_api,
            response_api=mock_response_api,
            ranking_table="binding_data",
            response_table="expression_data",
            ranking_score_column="binding_score",
            response_column="log2fc",
            comparison_id="test_comp",
            bin_size=5,
        )

        # Verify APIs were called correctly
        mock_ranking_api._ensure_dataset_loaded.assert_called()
        mock_response_api._ensure_dataset_loaded.assert_called()

        # Verify data was stored
        assert rank_response.db.table_exists("rank_response_test_comp")

        # Verify result
        assert not result.empty
        assert "regulator_locus_tag" in result.columns

    def test_compute_auto_generated_comparison_id(
        self, rank_response, mock_ranking_api, mock_response_api
    ):
        """Test compute with auto-generated comparison ID."""
        with patch("tfbpapi.HfRankResponse.duckdb.connect") as mock_duckdb:
            mock_temp_conn = MagicMock()
            mock_duckdb.return_value = mock_temp_conn
            mock_temp_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
                {
                    "regulator_locus_tag": ["REG1"],
                    "target_locus_tag": ["TGT1"],
                    "binding_score": [10.0],
                    "log2fc": [2.0],
                    "responsive": [1],
                    "bin_label": [5],
                    "cumulative_responsive": [1],
                }
            )

            rank_response.compute(
                ranking_api=mock_ranking_api,
                response_api=mock_response_api,
                ranking_table="binding_table",
                response_table="expression_table",
                ranking_score_column="binding_score",
                response_column="log2fc",
            )

            # Should create table with auto-generated ID
            assert rank_response.db.table_exists(
                "rank_response_binding_table_vs_expression_table"
            )

    def test_compute_incremental_update(
        self, rank_response, mock_ranking_api, mock_response_api, sample_computed_data
    ):
        """Test incremental computation with existing data."""
        # First, add some existing data
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        # Mock APIs to return new regulators
        mock_ranking_api.query.return_value = pd.DataFrame(
            {
                "regulator_locus_tag": ["REG1", "REG2", "REG3", "REG4"],  # REG4 is new
            }
        )

        with patch("tfbpapi.HfRankResponse.duckdb.connect") as mock_duckdb:
            mock_temp_conn = MagicMock()
            mock_duckdb.return_value = mock_temp_conn
            # Return data for new regulator only
            mock_temp_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
                {
                    "regulator_locus_tag": ["REG4"],
                    "target_locus_tag": ["TGT1"],
                    "binding_score": [5.0],
                    "log2fc": [1.0],
                    "responsive": [1],
                    "bin_label": [5],
                    "cumulative_responsive": [1],
                }
            )

            result = rank_response.compute(
                ranking_api=mock_ranking_api,
                response_api=mock_response_api,
                ranking_table="binding_data",
                response_table="expression_data",
                ranking_score_column="binding_score",
                response_column="log2fc",
                comparison_id="test_comp",
            )

            # Should have data for all regulators now
            regulators = set(result["regulator_locus_tag"].unique())
            assert "REG1" in regulators
            assert "REG4" in regulators

    def test_get_bin_summary(self, rank_response, sample_computed_data):
        """Test generating bin-level summary."""
        # Add test data
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        summary = rank_response.get_bin_summary("test_comp")

        assert not summary.empty
        assert "regulator_locus_tag" in summary.columns
        assert "bin_label" in summary.columns
        assert "records_in_bin" in summary.columns
        assert "responsive_in_bin" in summary.columns
        assert "cumulative_responsive" in summary.columns
        assert "response_rate" in summary.columns

    def test_get_bin_summary_with_filter(self, rank_response, sample_computed_data):
        """Test bin summary with regulator filter."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        summary = rank_response.get_bin_summary(
            "test_comp", regulators_filter=["REG1", "REG2"]
        )

        assert not summary.empty
        regulators = set(summary["regulator_locus_tag"].unique())
        assert regulators.issubset({"REG1", "REG2"})
        assert "REG3" not in regulators

    def test_get_regulator_summary(self, rank_response, sample_computed_data):
        """Test generating regulator-level summary."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        summary = rank_response.get_regulator_summary("test_comp")

        assert not summary.empty
        assert "regulator_locus_tag" in summary.columns
        assert "total_targets" in summary.columns
        assert "total_responsive" in summary.columns
        assert "overall_response_rate" in summary.columns
        assert "top5_response_rate" in summary.columns
        assert "top10_response_rate" in summary.columns
        assert "top20_response_rate" in summary.columns

    def test_get_regulator_summary_with_max_bin(
        self, rank_response, sample_computed_data
    ):
        """Test regulator summary with max bin limit."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        summary = rank_response.get_regulator_summary("test_comp", max_bin_label=5)

        assert not summary.empty
        # Should only include data from bins <= 5

    def test_summarize_bin_type(self, rank_response, sample_computed_data):
        """Test summarize method with bin type."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        summary = rank_response.summarize("test_comp", summary_type="bin")

        assert not summary.empty
        assert "bin_label" in summary.columns

    def test_summarize_regulator_type(self, rank_response, sample_computed_data):
        """Test summarize method with regulator type."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        summary = rank_response.summarize("test_comp", summary_type="regulator")

        assert not summary.empty
        assert "overall_response_rate" in summary.columns

    def test_summarize_invalid_type(self, rank_response, sample_computed_data):
        """Test summarize with invalid summary type."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        with pytest.raises(ValueError, match="Unknown summary type"):
            rank_response.summarize("test_comp", summary_type="invalid")

    def test_query_method(self, rank_response, sample_computed_data):
        """Test direct SQL query method."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        result = rank_response.query(
            "SELECT COUNT(*) as count FROM rank_response_test_comp"
        )

        assert not result.empty
        assert result.iloc[0]["count"] == len(sample_computed_data)

    def test_get_comparison_data(self, rank_response, sample_computed_data):
        """Test getting raw comparison data."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        data = rank_response.get_comparison_data("test_comp")

        assert not data.empty
        assert len(data) == len(sample_computed_data)

    def test_get_comparison_data_with_filter(self, rank_response, sample_computed_data):
        """Test getting comparison data with regulator filter."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        data = rank_response.get_comparison_data("test_comp", regulator_filter=["REG1"])

        assert not data.empty
        assert all(data["regulator_locus_tag"] == "REG1")

    def test_get_comparison_data_with_limit(self, rank_response, sample_computed_data):
        """Test getting comparison data with limit."""
        rank_response.db.append_results(sample_computed_data, "rank_response_test_comp")

        data = rank_response.get_comparison_data("test_comp", limit=3)

        assert len(data) == 3

    def test_compare_across_datasets(self, rank_response, sample_computed_data):
        """Test comparing across multiple datasets."""
        # Add data for multiple comparisons
        rank_response.db.append_results(sample_computed_data, "rank_response_comp1")
        rank_response.db.append_results(sample_computed_data, "rank_response_comp2")

        comparison = rank_response.compare_across_datasets(["comp1", "comp2"])

        assert not comparison.empty
        assert "regulator_locus_tag" in comparison.columns
        # Should have columns for each comparison and metric
        assert any("overall_response_rate_" in col for col in comparison.columns)

    def test_compare_across_datasets_empty(self, rank_response):
        """Test comparing across datasets with no data."""
        comparison = rank_response.compare_across_datasets([])

        assert comparison.empty

    def test_compare_across_datasets_custom_metrics(
        self, rank_response, sample_computed_data
    ):
        """Test comparing with custom metric columns."""
        rank_response.db.append_results(sample_computed_data, "rank_response_comp1")

        comparison = rank_response.compare_across_datasets(
            ["comp1"], metric_columns=["top5_response_rate"]
        )

        assert not comparison.empty
        assert any("top5_response_rate_" in col for col in comparison.columns)

    def test_nonexistent_comparison_error(self, rank_response):
        """Test error handling for nonexistent comparisons."""
        with pytest.raises(ValueError, match="No results found"):
            rank_response.get_comparison_data("nonexistent")

        with pytest.raises(ValueError, match="does not exist"):
            rank_response.get_bin_summary("nonexistent")

        with pytest.raises(ValueError, match="does not exist"):
            rank_response.get_regulator_summary("nonexistent")

    def test_compute_with_existing_filters(
        self, rank_response, mock_ranking_api, mock_response_api
    ):
        """Test compute when APIs already have filters set."""
        # Set existing filters
        mock_ranking_api.get_table_filter.return_value = "existing_filter = 'value'"
        mock_response_api.get_table_filter.return_value = "another_filter = 'value'"

        with patch("tfbpapi.HfRankResponse.duckdb.connect") as mock_duckdb:
            mock_temp_conn = MagicMock()
            mock_duckdb.return_value = mock_temp_conn
            mock_temp_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
                {
                    "regulator_locus_tag": ["REG1"],
                    "target_locus_tag": ["TGT1"],
                    "binding_score": [10.0],
                    "log2fc": [2.0],
                    "responsive": [1],
                    "bin_label": [5],
                    "cumulative_responsive": [1],
                }
            )

            rank_response.compute(
                ranking_api=mock_ranking_api,
                response_api=mock_response_api,
                ranking_table="binding_data",
                response_table="expression_data",
                ranking_score_column="binding_score",
                response_column="log2fc",
                comparison_id="with_filters",
            )

            # Verify filters were combined with AND
            calls = mock_ranking_api.set_table_filter.call_args_list
            assert len(calls) > 0
            combined_filter = calls[0][0][1]  # Second argument of first call
            assert "existing_filter = 'value'" in combined_filter
            assert "AND" in combined_filter

    @patch("tfbpapi.HfRankResponse.logging.getLogger")
    def test_logging_setup(self, mock_get_logger, analysis_db):
        """Test that logging is properly configured."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rr = HfRankResponse(analysis_db)

        mock_get_logger.assert_called_once_with("tfbpapi.HfRankResponse")
        assert rr.logger == mock_logger

    def test_compute_with_custom_responsive_condition(
        self, rank_response, mock_ranking_api, mock_response_api
    ):
        """Test compute with custom responsive condition."""
        with patch("tfbpapi.HfRankResponse.duckdb.connect") as mock_duckdb:
            mock_temp_conn = MagicMock()
            mock_duckdb.return_value = mock_temp_conn
            mock_temp_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
                {
                    "regulator_locus_tag": ["REG1"],
                    "target_locus_tag": ["TGT1"],
                    "binding_score": [10.0],
                    "log2fc": [2.0],
                    "responsive": [1],
                    "bin_label": [5],
                    "cumulative_responsive": [1],
                }
            )

            rank_response.compute(
                ranking_api=mock_ranking_api,
                response_api=mock_response_api,
                ranking_table="binding_data",
                response_table="expression_data",
                ranking_score_column="binding_score",
                response_column="log2fc",
                comparison_id="custom_responsive",
                responsive_condition="log2fc IS NOT NULL AND log2fc != 0",
            )

            # Verify the SQL contains the custom responsive condition
            sql_calls = mock_temp_conn.execute.call_args_list
            assert len(sql_calls) > 0
            executed_sql = sql_calls[0][0][0]
            assert "b.log2fc IS NOT NULL AND b.log2fc != 0" in executed_sql

    def test_compute_with_default_responsive_condition(
        self, rank_response, mock_ranking_api, mock_response_api
    ):
        """Test compute with default responsive condition (IS NOT NULL)."""
        with patch("tfbpapi.HfRankResponse.duckdb.connect") as mock_duckdb:
            mock_temp_conn = MagicMock()
            mock_duckdb.return_value = mock_temp_conn
            mock_temp_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
                {
                    "regulator_locus_tag": ["REG1"],
                    "target_locus_tag": ["TGT1"],
                    "binding_score": [10.0],
                    "log2fc": [2.0],
                    "responsive": [1],
                    "bin_label": [5],
                    "cumulative_responsive": [1],
                }
            )

            rank_response.compute(
                ranking_api=mock_ranking_api,
                response_api=mock_response_api,
                ranking_table="binding_data",
                response_table="expression_data",
                ranking_score_column="binding_score",
                response_column="log2fc",
                comparison_id="default_responsive",
            )

            # Verify the SQL contains the default responsive condition
            sql_calls = mock_temp_conn.execute.call_args_list
            assert len(sql_calls) > 0
            executed_sql = sql_calls[0][0][0]
            assert "b.log2fc IS NOT NULL" in executed_sql
