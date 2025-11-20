"""Tests for filters with automatic metadata joins."""

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
from tfbpapi.errors import InvalidFilterFieldError
from tfbpapi.HfQueryAPI import HfQueryAPI


@pytest.fixture
def mock_data_config():
    """Create a mock data configuration."""
    return DatasetConfig(
        config_name="annotated_features",
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
                    name="expression_value",
                    dtype="float64",
                    description="Expression value",
                ),
            ]
        ),
    )


@pytest.fixture
def mock_metadata_config():
    """Create a mock metadata configuration."""
    return DatasetConfig(
        config_name="sample_metadata",
        description="Test sample metadata",
        dataset_type=DatasetType.METADATA,
        applies_to=["annotated_features"],
        data_files=[DataFileInfo(path="metadata.parquet")],
        dataset_info=DatasetInfo(
            features=[
                FeatureInfo(
                    name="sample_id", dtype="string", description="Sample identifier"
                ),
                FeatureInfo(
                    name="data_usable", dtype="string", description="Data quality flag"
                ),
                FeatureInfo(name="cell_type", dtype="string", description="Cell type"),
            ]
        ),
    )


class TestFilterValidationWithMetadata:
    """Test that filter validation includes metadata columns."""

    def test_filter_on_metadata_field_validates(
        self, mock_data_config, mock_metadata_config
    ):
        """Test that filters on metadata fields pass validation."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        api._table_filters = {}

        # Mock get_config
        def get_config_side_effect(config_name):
            if config_name == "annotated_features":
                return mock_data_config
            elif config_name == "sample_metadata":
                return mock_metadata_config
            return None

        api.get_config = Mock(side_effect=get_config_side_effect)  # type: ignore

        # Mock get_metadata_relationships with inferred join key
        api.get_metadata_relationships = Mock(  # type: ignore
            return_value=[
                MetadataRelationship(
                    data_config="annotated_features",
                    metadata_config="sample_metadata",
                    relationship_type="explicit",
                    join_keys=["sample_id"],  # Inferred from column intersection
                )
            ]
        )

        # This should NOT raise - data_usable is in the metadata
        api._validate_metadata_fields("annotated_features", ["data_usable"])

    def test_filter_on_base_field_validates(
        self, mock_data_config, mock_metadata_config
    ):
        """Test that filters on base config fields still work."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        api._table_filters = {}

        # Mock get_config
        def get_config_side_effect(config_name):
            if config_name == "annotated_features":
                return mock_data_config
            elif config_name == "sample_metadata":
                return mock_metadata_config
            return None

        api.get_config = Mock(side_effect=get_config_side_effect)  # type: ignore

        # Mock get_metadata_relationships with inferred join key
        api.get_metadata_relationships = Mock(  # type: ignore
            return_value=[
                MetadataRelationship(
                    data_config="annotated_features",
                    metadata_config="sample_metadata",
                    relationship_type="explicit",
                    join_keys=["sample_id"],  # Inferred from column intersection
                )
            ]
        )

        # This should NOT raise - gene_id is in the base config
        api._validate_metadata_fields("annotated_features", ["gene_id"])

    def test_filter_on_invalid_field_fails(
        self, mock_data_config, mock_metadata_config
    ):
        """Test that filters on non-existent fields still fail."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        api._table_filters = {}

        # Mock get_config
        def get_config_side_effect(config_name):
            if config_name == "annotated_features":
                return mock_data_config
            elif config_name == "sample_metadata":
                return mock_metadata_config
            return None

        api.get_config = Mock(side_effect=get_config_side_effect)  # type: ignore

        # Mock get_metadata_relationships with inferred join key
        api.get_metadata_relationships = Mock(  # type: ignore
            return_value=[
                MetadataRelationship(
                    data_config="annotated_features",
                    metadata_config="sample_metadata",
                    relationship_type="explicit",
                    join_keys=["sample_id"],  # Inferred from column intersection
                )
            ]
        )

        # This SHOULD raise - nonexistent_field is nowhere
        with pytest.raises(InvalidFilterFieldError) as exc_info:
            api._validate_metadata_fields("annotated_features", ["nonexistent_field"])

        assert "nonexistent_field" in str(exc_info.value)

    def test_filter_validation_includes_both_sources(
        self, mock_data_config, mock_metadata_config
    ):
        """Test that validation includes fields from both base and metadata."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        api._table_filters = {}

        # Mock get_config
        def get_config_side_effect(config_name):
            if config_name == "annotated_features":
                return mock_data_config
            elif config_name == "sample_metadata":
                return mock_metadata_config
            return None

        api.get_config = Mock(side_effect=get_config_side_effect)  # type: ignore

        # Mock get_metadata_relationships with inferred join key
        api.get_metadata_relationships = Mock(  # type: ignore
            return_value=[
                MetadataRelationship(
                    data_config="annotated_features",
                    metadata_config="sample_metadata",
                    relationship_type="explicit",
                    join_keys=["sample_id"],  # Inferred from column intersection
                )
            ]
        )

        # Mix of base and metadata fields should all validate
        api._validate_metadata_fields(
            "annotated_features", ["gene_id", "data_usable", "cell_type"]
        )


class TestFilterAutoJoinTrigger:
    """Test that filters trigger automatic metadata joins."""

    def test_stored_filter_triggers_join(self, mock_data_config, mock_metadata_config):
        """Test that stored filters are analyzed for metadata columns."""
        api = HfQueryAPI.__new__(HfQueryAPI)
        api.logger = MagicMock()
        api._table_filters = {"annotated_features": "data_usable = 'pass'"}

        # Mock get_config
        def get_config_side_effect(config_name):
            if config_name == "annotated_features":
                return mock_data_config
            elif config_name == "sample_metadata":
                return mock_metadata_config
            return None

        api.get_config = Mock(side_effect=get_config_side_effect)  # type: ignore

        # Mock get_metadata_relationships with inferred join key
        api.get_metadata_relationships = Mock(  # type: ignore
            return_value=[
                MetadataRelationship(
                    data_config="annotated_features",
                    metadata_config="sample_metadata",
                    relationship_type="explicit",
                    join_keys=["sample_id"],  # Inferred from column intersection
                )
            ]
        )

        # When extracting columns from a simple query, the filter should also be checked
        # Simulating the query flow
        sql = "SELECT * FROM annotated_features"

        # Extract from query
        referenced_columns = api._extract_column_references(sql)

        # Extract from filter
        if "annotated_features" in api._table_filters:
            filter_sql = api._table_filters["annotated_features"]
            filter_columns = api._extract_column_references(filter_sql)
            referenced_columns.update(filter_columns)

        # Should include data_usable from the filter
        assert "data_usable" in referenced_columns

        # Check that metadata would be found
        base_columns = api._get_columns_from_config("annotated_features")
        missing_columns = referenced_columns - base_columns

        assert "data_usable" in missing_columns

        # Verify it would find the metadata
        metadata_matches = api._find_metadata_for_columns(
            "annotated_features", missing_columns
        )

        assert len(metadata_matches) == 1
        assert metadata_matches[0][0] == "sample_metadata"
