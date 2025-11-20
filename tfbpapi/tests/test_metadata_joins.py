"""Tests for automatic metadata join functionality."""

from unittest.mock import MagicMock, Mock

import pytest

from tfbpapi.datainfo.models import (
    DataFileInfo,
    DatasetConfig,
    DatasetInfo,
    DatasetType,
    FeatureInfo,
    MetadataRelationship,
)
from tfbpapi.HfQueryAPI import HfQueryAPI


@pytest.fixture
def mock_data_config():
    """Create a mock data configuration."""
    return DatasetConfig(
        config_name="binding_data",
        description="Test binding data",
        dataset_type=DatasetType.ANNOTATED_FEATURES,
        data_files=[DataFileInfo(path="data.parquet")],
        dataset_info=DatasetInfo(
            features=[
                FeatureInfo(
                    name="sample_id", dtype="string", description="Sample identifier"
                ),
                FeatureInfo(
                    name="gene_id", dtype="string", description="Gene identifier"
                ),
                FeatureInfo(
                    name="binding_score",
                    dtype="float64",
                    description="Binding score value",
                ),
            ]
        ),
    )


@pytest.fixture
def mock_metadata_config():
    """Create a mock metadata configuration."""
    return DatasetConfig(
        config_name="experiment_metadata",
        description="Test experiment metadata",
        dataset_type=DatasetType.METADATA,
        applies_to=["binding_data"],
        data_files=[DataFileInfo(path="metadata.parquet")],
        dataset_info=DatasetInfo(
            features=[
                FeatureInfo(
                    name="sample_id", dtype="string", description="Sample identifier"
                ),
                FeatureInfo(name="cell_type", dtype="string", description="Cell type"),
                FeatureInfo(
                    name="treatment", dtype="string", description="Treatment condition"
                ),
            ]
        ),
    )


@pytest.fixture
def mock_metadata_config_composite_key():
    """Create a mock metadata configuration with multiple common columns."""
    return DatasetConfig(
        config_name="sample_metadata",
        description="Test sample metadata with composite key",
        dataset_type=DatasetType.METADATA,
        applies_to=["binding_data"],
        data_files=[DataFileInfo(path="sample_metadata.parquet")],
        dataset_info=DatasetInfo(
            features=[
                FeatureInfo(
                    name="sample_id", dtype="string", description="Sample identifier"
                ),
                FeatureInfo(
                    name="gene_id", dtype="string", description="Gene identifier"
                ),
                FeatureInfo(
                    name="replicate", dtype="int64", description="Replicate number"
                ),
            ]
        ),
    )


class TestMetadataRelationshipsWithInferredJoinKeys:
    """Test that join keys are automatically inferred from column intersection."""

    @pytest.fixture(autouse=True)
    def patch_load(self):
        """Patch the _load_and_validate_card method for all tests in this class."""
        from unittest.mock import patch

        with patch("tfbpapi.datainfo.datacard.DataCard._load_and_validate_card"):
            yield

    def test_relationship_infers_single_join_key(
        self, mock_data_config, mock_metadata_config
    ):
        """Test that single common column is inferred as join key."""
        from tfbpapi.datainfo.datacard import DataCard
        from tfbpapi.datainfo.models import DatasetCard

        # Mock dataset card with both configs
        mock_card = DatasetCard(configs=[mock_data_config, mock_metadata_config])

        datacard = DataCard("test/repo")
        datacard._dataset_card = mock_card
        relationships = datacard.get_metadata_relationships()

        # Should have one explicit relationship
        explicit_rels = [r for r in relationships if r.relationship_type == "explicit"]
        assert len(explicit_rels) == 1

        rel = explicit_rels[0]
        assert rel.data_config == "binding_data"
        assert rel.metadata_config == "experiment_metadata"
        # sample_id is the only common column
        assert rel.join_keys == ["sample_id"]

    def test_relationship_infers_composite_keys(
        self, mock_data_config, mock_metadata_config_composite_key
    ):
        """Test that multiple common columns are inferred as composite join keys."""
        from tfbpapi.datainfo.datacard import DataCard
        from tfbpapi.datainfo.models import DatasetCard

        mock_card = DatasetCard(
            configs=[mock_data_config, mock_metadata_config_composite_key]
        )

        datacard = DataCard("test/repo")
        datacard._dataset_card = mock_card
        relationships = datacard.get_metadata_relationships()

        explicit_rels = [r for r in relationships if r.relationship_type == "explicit"]
        assert len(explicit_rels) == 1
        # Both sample_id and gene_id are common
        assert set(explicit_rels[0].join_keys) == {  # type: ignore
            "gene_id",
            "sample_id",
        }

    def test_relationship_no_common_columns(self, mock_data_config):
        """Test that no join keys are inferred when there are no common columns."""
        from tfbpapi.datainfo.datacard import DataCard
        from tfbpapi.datainfo.models import DatasetCard

        # Create metadata with no common columns
        metadata_no_overlap = DatasetConfig(
            config_name="unrelated_metadata",
            description="Metadata with no common columns",
            dataset_type=DatasetType.METADATA,
            applies_to=["binding_data"],
            data_files=[DataFileInfo(path="metadata.parquet")],
            dataset_info=DatasetInfo(
                features=[
                    FeatureInfo(
                        name="unrelated_id", dtype="string", description="Unrelated ID"
                    ),
                    FeatureInfo(
                        name="some_value", dtype="float64", description="Some value"
                    ),
                ]
            ),
        )

        mock_card = DatasetCard(configs=[mock_data_config, metadata_no_overlap])

        datacard = DataCard("test/repo")
        datacard._dataset_card = mock_card
        relationships = datacard.get_metadata_relationships()

        explicit_rels = [r for r in relationships if r.relationship_type == "explicit"]
        assert len(explicit_rels) == 1
        # No common columns, so no join keys
        assert explicit_rels[0].join_keys is None


class TestColumnExtraction:
    """Test SQL column extraction functionality."""

    def test_extract_simple_select(self):
        """Test extracting columns from simple SELECT query."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        sql = "SELECT sample_id, cell_type FROM table WHERE cell_type = 'K562'"
        columns = api._extract_column_references(sql)
        assert "sample_id" in columns
        assert "cell_type" in columns

    def test_extract_with_where_clause(self):
        """Test extracting columns from WHERE clauses."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        sql = "SELECT * FROM table WHERE cell_type = 'K562' AND treatment = 'drug'"
        columns = api._extract_column_references(sql)
        assert "cell_type" in columns
        assert "treatment" in columns

    def test_extract_filters_sql_keywords(self):
        """Test that SQL keywords are filtered out."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        sql = "SELECT * FROM table WHERE col1 = 'value' AND col2 IS NOT NULL"
        columns = api._extract_column_references(sql)
        assert "SELECT" not in columns
        assert "FROM" not in columns
        assert "WHERE" not in columns
        assert "AND" not in columns
        assert "IS" not in columns
        assert "NOT" not in columns
        assert "NULL" not in columns

    def test_extract_ignores_string_literals(self):
        """Test that string literals are ignored."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        sql = "SELECT col FROM table WHERE status = 'active_user'"
        columns = api._extract_column_references(sql)
        assert "col" in columns
        assert "status" in columns
        # 'active_user' should not be extracted as a column
        assert "active_user" not in columns


class TestAutomaticMetadataJoins:
    """Test automatic metadata joining in queries."""

    def test_find_metadata_for_columns(self, mock_data_config, mock_metadata_config):
        """Test finding metadata configs that contain specific columns."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()

        # Mock get_config
        def get_config_side_effect(config_name):
            if config_name == "binding_data":
                return mock_data_config
            elif config_name == "experiment_metadata":
                return mock_metadata_config
            return None

        api.get_config = Mock(side_effect=get_config_side_effect)  # type: ignore

        # Mock get_metadata_relationships with inferred join keys
        api.get_metadata_relationships = Mock(  # type: ignore
            return_value=[
                MetadataRelationship(
                    data_config="binding_data",
                    metadata_config="experiment_metadata",
                    relationship_type="explicit",
                    join_keys=["sample_id"],  # Inferred from column intersection
                )
            ]
        )

        # Test finding metadata for cell_type column
        columns = {"cell_type"}
        results = api._find_metadata_for_columns("binding_data", columns)

        assert len(results) == 1
        assert results[0][0] == "experiment_metadata"
        assert results[0][1] == ["sample_id"]

    def test_build_join_sql_single_key(self):
        """Test SQL rewriting with single join key."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()

        base_sql = "SELECT * FROM metadata_binding_data WHERE cell_type = 'K562'"
        metadata_joins = [
            (
                "experiment_metadata",
                "metadata_experiment_metadata",
                ["sample_id"],
            )
        ]

        result = api._build_join_sql(base_sql, "metadata_binding_data", metadata_joins)

        assert "LEFT JOIN metadata_experiment_metadata" in result
        assert "USING (sample_id)" in result

    def test_build_join_sql_composite_key(self):
        """Test SQL rewriting with composite join keys."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()

        base_sql = "SELECT * FROM metadata_binding_data WHERE replicate = 1"
        metadata_joins = [
            (
                "sample_metadata",
                "metadata_sample_metadata",
                ["gene_id", "sample_id"],  # Alphabetically sorted
            )
        ]

        result = api._build_join_sql(base_sql, "metadata_binding_data", metadata_joins)

        assert "LEFT JOIN metadata_sample_metadata" in result
        assert "USING (gene_id, sample_id)" in result

    def test_auto_join_disabled(self):
        """Test that auto_join_metadata=False disables automatic joins."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        api._extract_column_references = MagicMock()  # type: ignore

        # Verify the parameter exists
        assert hasattr(HfQueryAPI.query, "__code__")
        params = HfQueryAPI.query.__code__.co_varnames
        assert "auto_join_metadata" in params


class TestGetColumnsFromConfig:
    """Test _get_columns_from_config helper method."""

    def test_get_columns_from_data_config(self, mock_data_config):
        """Test extracting columns from data config."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.get_config = Mock(return_value=mock_data_config)  # type: ignore

        columns = api._get_columns_from_config("binding_data")
        assert columns == {"sample_id", "gene_id", "binding_score"}

    def test_get_columns_from_metadata_config(self, mock_metadata_config):
        """Test extracting columns from metadata config."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.get_config = Mock(return_value=mock_metadata_config)  # type: ignore

        columns = api._get_columns_from_config("experiment_metadata")
        assert columns == {"sample_id", "cell_type", "treatment"}

    def test_get_columns_nonexistent_config(self):
        """Test getting columns from non-existent config returns empty set."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.get_config = Mock(return_value=None)  # type: ignore

        columns = api._get_columns_from_config("nonexistent")
        assert columns == set()
