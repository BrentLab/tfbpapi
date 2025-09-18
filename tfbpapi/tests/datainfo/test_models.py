"""Tests for datainfo Pydantic models."""

import pytest
from pydantic import ValidationError

from tfbpapi.datainfo.models import (
    DataFileInfo,
    DatasetCard,
    DatasetConfig,
    DatasetInfo,
    DatasetType,
    ExtractedMetadata,
    FeatureInfo,
    MetadataRelationship,
    PartitioningInfo,
)


class TestDatasetType:
    """Test DatasetType enum."""

    def test_dataset_type_values(self):
        """Test all dataset type enum values."""
        assert DatasetType.GENOMIC_FEATURES == "genomic_features"
        assert DatasetType.ANNOTATED_FEATURES == "annotated_features"
        assert DatasetType.GENOME_MAP == "genome_map"
        assert DatasetType.METADATA == "metadata"

    def test_dataset_type_from_string(self):
        """Test creating DatasetType from string."""
        assert DatasetType("genomic_features") == DatasetType.GENOMIC_FEATURES
        assert DatasetType("annotated_features") == DatasetType.ANNOTATED_FEATURES
        assert DatasetType("genome_map") == DatasetType.GENOME_MAP

    def test_invalid_dataset_type(self):
        """Test invalid dataset type raises error."""
        with pytest.raises(ValueError):
            DatasetType("invalid_type")


class TestFeatureInfo:
    """Test FeatureInfo model."""

    def test_valid_feature_info(self, sample_feature_info):
        """Test creating valid FeatureInfo."""
        feature = FeatureInfo(**sample_feature_info)
        assert feature.name == "gene_symbol"
        assert feature.dtype == "string"
        assert feature.description == "Standard gene symbol (e.g., HO, GAL1)"

    def test_feature_info_required_fields(self):
        """Test that all fields are required."""
        # Test missing name
        with pytest.raises(ValidationError):
            FeatureInfo(dtype="string", description="test")

        # Test missing dtype
        with pytest.raises(ValidationError):
            FeatureInfo(name="test", description="test")

        # Test missing description
        with pytest.raises(ValidationError):
            FeatureInfo(name="test", dtype="string")

    def test_feature_info_serialization(self, sample_feature_info):
        """Test FeatureInfo serialization."""
        feature = FeatureInfo(**sample_feature_info)
        data = feature.model_dump()
        assert data["name"] == "gene_symbol"
        assert data["dtype"] == "string"
        assert data["description"] == "Standard gene symbol (e.g., HO, GAL1)"

    def test_feature_info_categorical_dtype(self):
        """Test FeatureInfo with categorical dtype."""
        categorical_feature_data = {
            "name": "mechanism",
            "dtype": {"class_label": {"names": ["GEV", "ZEV"]}},
            "description": "Induction system",
        }

        feature = FeatureInfo(**categorical_feature_data)
        assert feature.name == "mechanism"
        assert isinstance(feature.dtype, dict)
        assert "class_label" in feature.dtype
        assert feature.dtype["class_label"].names == ["GEV", "ZEV"]
        assert feature.get_dtype_summary() == "categorical (2 classes: GEV, ZEV)"

    def test_feature_info_simple_string_dtype(self):
        """Test FeatureInfo with simple string dtype."""
        feature = FeatureInfo(
            name="test_field", dtype="string", description="Test field"
        )
        assert feature.dtype == "string"
        assert feature.get_dtype_summary() == "string"

    def test_feature_info_invalid_categorical_dtype(self):
        """Test FeatureInfo with invalid categorical dtype structure."""
        with pytest.raises(ValidationError, match="Invalid class_label structure"):
            FeatureInfo(
                name="test_field",
                dtype={"class_label": "invalid"},  # Should be dict with names
                description="Test field",
            )

    def test_feature_info_unknown_dtype_structure(self):
        """Test FeatureInfo with unknown dtype structure."""
        with pytest.raises(ValidationError, match="Unknown dtype structure"):
            FeatureInfo(
                name="test_field",
                dtype={"unknown_key": "value"},
                description="Test field",
            )


class TestPartitioningInfo:
    """Test PartitioningInfo model."""

    def test_default_partitioning_info(self):
        """Test default PartitioningInfo values."""
        partitioning = PartitioningInfo()
        assert partitioning.enabled is False
        assert partitioning.partition_by is None
        assert partitioning.path_template is None

    def test_enabled_partitioning_info(self, sample_partitioning_info):
        """Test enabled partitioning with all fields."""
        partitioning = PartitioningInfo(**sample_partitioning_info)
        assert partitioning.enabled is True
        assert partitioning.partition_by == ["regulator", "condition"]
        assert (
            partitioning.path_template is not None
            and "regulator={regulator}" in partitioning.path_template
        )

    def test_partial_partitioning_info(self):
        """Test partitioning with only some fields set."""
        partitioning = PartitioningInfo(enabled=True, partition_by=["field1"])
        assert partitioning.enabled is True
        assert partitioning.partition_by == ["field1"]
        assert partitioning.path_template is None


class TestDataFileInfo:
    """Test DataFileInfo model."""

    def test_default_data_file_info(self):
        """Test DataFileInfo with default split."""
        data_file = DataFileInfo(path="test.parquet")
        assert data_file.split == "train"
        assert data_file.path == "test.parquet"

    def test_custom_data_file_info(self, sample_data_file_info):
        """Test DataFileInfo with custom values."""
        data_file = DataFileInfo(**sample_data_file_info)
        assert data_file.split == "train"
        assert data_file.path == "genomic_features.parquet"

    def test_data_file_info_required_path(self):
        """Test that path is required."""
        with pytest.raises(ValidationError):
            DataFileInfo(split="train")


class TestDatasetInfo:
    """Test DatasetInfo model."""

    def test_minimal_dataset_info(self, sample_feature_info):
        """Test minimal DatasetInfo with just features."""
        features = [FeatureInfo(**sample_feature_info)]
        dataset_info = DatasetInfo(features=features)
        assert len(dataset_info.features) == 1
        assert dataset_info.partitioning is None

    def test_dataset_info_with_partitioning(
        self, sample_feature_info, sample_partitioning_info
    ):
        """Test DatasetInfo with partitioning."""
        features = [FeatureInfo(**sample_feature_info)]
        partitioning = PartitioningInfo(**sample_partitioning_info)

        dataset_info = DatasetInfo(features=features, partitioning=partitioning)
        assert len(dataset_info.features) == 1
        assert (
            dataset_info.partitioning is not None
            and dataset_info.partitioning.enabled is True
        )

    def test_dataset_info_empty_features_error(self):
        """Test that empty features list is allowed."""
        # Pydantic allows empty lists, so this should succeed
        dataset_info = DatasetInfo(features=[])
        assert len(dataset_info.features) == 0


class TestDatasetConfig:
    """Test DatasetConfig model."""

    def test_minimal_dataset_config(self, sample_feature_info, sample_data_file_info):
        """Test minimal valid DatasetConfig."""
        features = [FeatureInfo(**sample_feature_info)]
        data_files = [DataFileInfo(**sample_data_file_info)]
        dataset_info = DatasetInfo(features=features)

        config = DatasetConfig(
            config_name="test_config",
            description="Test configuration",
            dataset_type=DatasetType.GENOMIC_FEATURES,
            data_files=data_files,
            dataset_info=dataset_info,
        )

        assert config.config_name == "test_config"
        assert config.dataset_type == DatasetType.GENOMIC_FEATURES
        assert config.default is False
        assert config.applies_to is None
        assert config.metadata_fields is None

    def test_dataset_config_with_applies_to_metadata(
        self, sample_feature_info, sample_data_file_info
    ):
        """Test DatasetConfig with applies_to for metadata types."""
        features = [FeatureInfo(**sample_feature_info)]
        data_files = [DataFileInfo(**sample_data_file_info)]
        dataset_info = DatasetInfo(features=features)

        config = DatasetConfig(
            config_name="metadata_config",
            description="Metadata configuration",
            dataset_type=DatasetType.METADATA,
            applies_to=["data_config1", "data_config2"],
            data_files=data_files,
            dataset_info=dataset_info,
        )

        assert config.applies_to == ["data_config1", "data_config2"]

    def test_dataset_config_applies_to_validation_error(
        self, sample_feature_info, sample_data_file_info
    ):
        """Test that applies_to is only valid for metadata types."""
        features = [FeatureInfo(**sample_feature_info)]
        data_files = [DataFileInfo(**sample_data_file_info)]
        dataset_info = DatasetInfo(features=features)

        with pytest.raises(
            ValidationError,
            match="applies_to field is only valid for metadata dataset types",
        ):
            DatasetConfig(
                config_name="invalid_config",
                description="Invalid configuration",
                dataset_type=DatasetType.GENOMIC_FEATURES,  # Not a metadata type
                applies_to=["some_config"],  # This should cause validation error
                data_files=data_files,
                dataset_info=dataset_info,
            )

    def test_dataset_config_empty_metadata_fields_error(
        self, sample_feature_info, sample_data_file_info
    ):
        """Test that empty metadata_fields list raises error."""
        features = [FeatureInfo(**sample_feature_info)]
        data_files = [DataFileInfo(**sample_data_file_info)]
        dataset_info = DatasetInfo(features=features)

        with pytest.raises(
            ValidationError, match="metadata_fields cannot be empty list"
        ):
            DatasetConfig(
                config_name="test_config",
                description="Test configuration",
                dataset_type=DatasetType.ANNOTATED_FEATURES,
                metadata_fields=[],  # Empty list should cause error
                data_files=data_files,
                dataset_info=dataset_info,
            )

    def test_dataset_config_with_metadata_fields(
        self, sample_feature_info, sample_data_file_info
    ):
        """Test DatasetConfig with valid metadata_fields."""
        features = [FeatureInfo(**sample_feature_info)]
        data_files = [DataFileInfo(**sample_data_file_info)]
        dataset_info = DatasetInfo(features=features)

        config = DatasetConfig(
            config_name="test_config",
            description="Test configuration",
            dataset_type=DatasetType.ANNOTATED_FEATURES,
            metadata_fields=["field1", "field2"],
            data_files=data_files,
            dataset_info=dataset_info,
        )

        assert config.metadata_fields == ["field1", "field2"]


class TestDatasetCard:
    """Test DatasetCard model."""

    def test_minimal_dataset_card(self, minimal_dataset_card_data):
        """Test minimal valid DatasetCard."""
        card = DatasetCard(**minimal_dataset_card_data)
        assert len(card.configs) == 1
        assert card.configs[0].config_name == "test_config"
        assert card.license is None
        assert card.pretty_name is None

    def test_full_dataset_card(self, sample_dataset_card_data):
        """Test full DatasetCard with all fields."""
        card = DatasetCard(**sample_dataset_card_data)
        assert len(card.configs) == 4
        assert card.license == "mit"
        assert card.pretty_name == "Test Genomics Dataset"
        assert card.tags is not None and "biology" in card.tags

    def test_empty_configs_error(self):
        """Test that empty configs list raises error."""
        with pytest.raises(
            ValidationError, match="At least one dataset configuration is required"
        ):
            DatasetCard(configs=[])

    def test_duplicate_config_names_error(
        self, sample_feature_info, sample_data_file_info
    ):
        """Test that duplicate config names raise error."""
        features = [FeatureInfo(**sample_feature_info)]
        data_files = [DataFileInfo(**sample_data_file_info)]
        dataset_info = DatasetInfo(features=features)

        config1 = DatasetConfig(
            config_name="duplicate_name",
            description="First config",
            dataset_type=DatasetType.GENOMIC_FEATURES,
            data_files=data_files,
            dataset_info=dataset_info,
        )

        config2 = DatasetConfig(
            config_name="duplicate_name",  # Same name
            description="Second config",
            dataset_type=DatasetType.ANNOTATED_FEATURES,
            data_files=data_files,
            dataset_info=dataset_info,
        )

        with pytest.raises(ValidationError, match="Configuration names must be unique"):
            DatasetCard(configs=[config1, config2])

    def test_multiple_default_configs_error(
        self, sample_feature_info, sample_data_file_info
    ):
        """Test that multiple default configs raise error."""
        features = [FeatureInfo(**sample_feature_info)]
        data_files = [DataFileInfo(**sample_data_file_info)]
        dataset_info = DatasetInfo(features=features)

        config1 = DatasetConfig(
            config_name="config1",
            description="First config",
            dataset_type=DatasetType.GENOMIC_FEATURES,
            default=True,
            data_files=data_files,
            dataset_info=dataset_info,
        )

        config2 = DatasetConfig(
            config_name="config2",
            description="Second config",
            dataset_type=DatasetType.ANNOTATED_FEATURES,
            default=True,  # Another default
            data_files=data_files,
            dataset_info=dataset_info,
        )

        with pytest.raises(
            ValidationError, match="At most one configuration can be marked as default"
        ):
            DatasetCard(configs=[config1, config2])

    def test_get_config_by_name(self, sample_dataset_card_data):
        """Test getting config by name."""
        card = DatasetCard(**sample_dataset_card_data)

        config = card.get_config_by_name("binding_data")
        assert config is not None
        assert config.config_name == "binding_data"

        # Test non-existent config
        assert card.get_config_by_name("nonexistent") is None

    def test_get_configs_by_type(self, sample_dataset_card_data):
        """Test getting configs by type."""
        card = DatasetCard(**sample_dataset_card_data)

        genomic_configs = card.get_configs_by_type(DatasetType.GENOMIC_FEATURES)
        assert len(genomic_configs) == 1
        assert genomic_configs[0].config_name == "genomic_features"

        metadata_configs = card.get_configs_by_type(DatasetType.METADATA)
        assert len(metadata_configs) == 1
        assert metadata_configs[0].config_name == "experiment_metadata"

    def test_get_default_config(self, sample_dataset_card_data):
        """Test getting default config."""
        card = DatasetCard(**sample_dataset_card_data)

        default_config = card.get_default_config()
        assert default_config is not None
        assert default_config.config_name == "genomic_features"
        assert default_config.default is True

    def test_get_data_configs(self, sample_dataset_card_data):
        """Test getting non-metadata configs."""
        card = DatasetCard(**sample_dataset_card_data)

        data_configs = card.get_data_configs()
        assert len(data_configs) == 3  # genomic_features, binding_data, genome_map_data
        config_names = [config.config_name for config in data_configs]
        assert "genomic_features" in config_names
        assert "binding_data" in config_names
        assert "genome_map_data" in config_names
        assert "experiment_metadata" not in config_names

    def test_get_metadata_configs(self, sample_dataset_card_data):
        """Test getting metadata configs."""
        card = DatasetCard(**sample_dataset_card_data)

        metadata_configs = card.get_metadata_configs()
        assert len(metadata_configs) == 1
        assert metadata_configs[0].config_name == "experiment_metadata"


class TestExtractedMetadata:
    """Test ExtractedMetadata model."""

    def test_extracted_metadata_creation(self):
        """Test creating ExtractedMetadata."""
        metadata = ExtractedMetadata(
            config_name="test_config",
            field_name="regulator_symbol",
            values={"TF1", "TF2", "TF3"},
            extraction_method="partition_values",
        )

        assert metadata.config_name == "test_config"
        assert metadata.field_name == "regulator_symbol"
        assert metadata.values == {"TF1", "TF2", "TF3"}
        assert metadata.extraction_method == "partition_values"

    def test_extracted_metadata_serialization(self):
        """Test ExtractedMetadata JSON serialization."""
        metadata = ExtractedMetadata(
            config_name="test_config",
            field_name="condition",
            values={"control", "treatment"},
            extraction_method="embedded",
        )

        # Test basic serialization (sets remain as sets in model_dump)
        data = metadata.model_dump()
        assert isinstance(data["values"], set)
        assert data["values"] == {"control", "treatment"}

        # Test JSON mode serialization where sets should become lists
        json_data = metadata.model_dump(mode="json")
        assert isinstance(json_data["values"], list)
        assert set(json_data["values"]) == {"control", "treatment"}


class TestMetadataRelationship:
    """Test MetadataRelationship model."""

    def test_metadata_relationship_creation(self):
        """Test creating MetadataRelationship."""
        relationship = MetadataRelationship(
            data_config="binding_data",
            metadata_config="experiment_metadata",
            relationship_type="explicit",
        )

        assert relationship.data_config == "binding_data"
        assert relationship.metadata_config == "experiment_metadata"
        assert relationship.relationship_type == "explicit"

    def test_metadata_relationship_types(self):
        """Test different relationship types."""
        # Test explicit relationship
        explicit = MetadataRelationship(
            data_config="data1", metadata_config="meta1", relationship_type="explicit"
        )
        assert explicit.relationship_type == "explicit"

        # Test embedded relationship
        embedded = MetadataRelationship(
            data_config="data3",
            metadata_config="data3_embedded",
            relationship_type="embedded",
        )
        assert embedded.relationship_type == "embedded"
