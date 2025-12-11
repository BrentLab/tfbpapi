"""Tests for the DataCard class."""

from unittest.mock import Mock, patch

import pytest

from tfbpapi.datainfo import DataCard
from tfbpapi.datainfo.models import DatasetType
from tfbpapi.errors import DataCardError, DataCardValidationError, HfDataFetchError


class TestDataCard:
    """Test suite for DataCard class."""

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_init(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        test_token,
    ):
        """Test DataCard initialization."""
        datacard = DataCard(test_repo_id, token=test_token)

        assert datacard.repo_id == test_repo_id
        assert datacard.token == test_token
        assert datacard._dataset_card is None
        assert datacard._metadata_cache == {}

        # Check that fetchers were initialized
        mock_card_fetcher.assert_called_once_with(token=test_token)
        mock_structure_fetcher.assert_called_once_with(token=test_token)
        mock_size_fetcher.assert_called_once_with(token=test_token)

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_init_without_token(
        self, mock_size_fetcher, mock_structure_fetcher, mock_card_fetcher, test_repo_id
    ):
        """Test DataCard initialization without token."""
        datacard = DataCard(test_repo_id)

        assert datacard.repo_id == test_repo_id
        assert datacard.token is None

        # Check that fetchers were initialized without token
        mock_card_fetcher.assert_called_once_with(token=None)
        mock_structure_fetcher.assert_called_once_with(token=None)
        mock_size_fetcher.assert_called_once_with(token=None)

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_load_and_validate_card_success(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test successful card loading and validation."""
        # Setup mock
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        # Access dataset_card property to trigger loading
        card = datacard.dataset_card

        assert card is not None
        assert len(card.configs) == 4
        assert card.pretty_name == "Test Genomics Dataset"
        mock_fetcher_instance.fetch.assert_called_once_with(test_repo_id)

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_load_card_no_data(
        self, mock_size_fetcher, mock_structure_fetcher, mock_card_fetcher, test_repo_id
    ):
        """Test handling when no dataset card is found."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = {}

        datacard = DataCard(test_repo_id)

        with pytest.raises(DataCardValidationError, match="No dataset card found"):
            _ = datacard.dataset_card

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_load_card_validation_error(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        invalid_dataset_card_data,
    ):
        """Test handling of validation errors."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = invalid_dataset_card_data

        datacard = DataCard(test_repo_id)

        with pytest.raises(
            DataCardValidationError, match="Dataset card validation failed"
        ):
            _ = datacard.dataset_card

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_load_card_fetch_error(
        self, mock_size_fetcher, mock_structure_fetcher, mock_card_fetcher, test_repo_id
    ):
        """Test handling of fetch errors."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.side_effect = HfDataFetchError("Fetch failed")

        datacard = DataCard(test_repo_id)

        with pytest.raises(DataCardError, match="Failed to fetch dataset card"):
            _ = datacard.dataset_card

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_configs_property(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test getting all configurations via property."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)
        configs = datacard.configs

        assert len(configs) == 4
        config_names = [config.config_name for config in configs]
        assert "genomic_features" in config_names
        assert "binding_data" in config_names
        assert "genome_map_data" in config_names
        assert "experiment_metadata" in config_names

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_config_by_name(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test getting a specific configuration by name."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        config = datacard.get_config("binding_data")
        assert config is not None
        assert config.config_name == "binding_data"
        assert config.dataset_type == DatasetType.ANNOTATED_FEATURES

        # Test non-existent config
        assert datacard.get_config("nonexistent") is None

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_configs_by_type(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test getting configurations by dataset type."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        # Test with enum
        genomic_configs = datacard.get_configs_by_type(DatasetType.GENOMIC_FEATURES)
        assert len(genomic_configs) == 1
        assert genomic_configs[0].config_name == "genomic_features"

        # Test with string
        metadata_configs = datacard.get_configs_by_type("metadata")
        assert len(metadata_configs) == 1
        assert metadata_configs[0].config_name == "experiment_metadata"

        # Test with genome_map type
        genome_map_configs = datacard.get_configs_by_type("genome_map")
        assert len(genome_map_configs) == 1
        assert genome_map_configs[0].config_name == "genome_map_data"

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_field_values_success(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test getting field values for a specific config and field."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        # Test existing field
        values = datacard.get_field_values("binding_data", "regulator_symbol")
        # Since _extract_field_values returns empty set by default, we expect empty set
        assert isinstance(values, set)

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_field_values_config_not_found(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test error when config not found."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        with pytest.raises(
            DataCardError, match="Configuration 'nonexistent' not found"
        ):
            datacard.get_field_values("nonexistent", "some_field")

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_field_values_field_not_found(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test error when field not found."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        with pytest.raises(DataCardError, match="Field 'nonexistent' not found"):
            datacard.get_field_values("binding_data", "nonexistent")

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_metadata_relationships(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test getting metadata relationships."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        relationships = datacard.get_metadata_relationships()

        # Should have explicit relationship between binding_data and experiment_metadata
        explicit_rels = [r for r in relationships if r.relationship_type == "explicit"]
        assert len(explicit_rels) == 1
        assert explicit_rels[0].data_config == "binding_data"
        assert explicit_rels[0].metadata_config == "experiment_metadata"

        # Should have embedded relationship for binding_data (has metadata_fields)
        embedded_rels = [r for r in relationships if r.relationship_type == "embedded"]
        assert len(embedded_rels) == 1
        assert embedded_rels[0].data_config == "binding_data"
        assert embedded_rels[0].metadata_config == "binding_data_embedded"

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_repository_info_success(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
        sample_repo_structure,
    ):
        """Test getting repository information."""
        mock_card_fetcher_instance = Mock()
        mock_structure_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_structure_fetcher.return_value = mock_structure_fetcher_instance

        mock_card_fetcher_instance.fetch.return_value = sample_dataset_card_data
        mock_structure_fetcher_instance.fetch.return_value = sample_repo_structure

        datacard = DataCard(test_repo_id)

        info = datacard.get_repository_info()

        assert info["repo_id"] == test_repo_id
        assert info["pretty_name"] == "Test Genomics Dataset"
        assert info["license"] == "mit"
        assert info["num_configs"] == 4
        assert "genomic_features" in info["dataset_types"]
        assert "annotated_features" in info["dataset_types"]
        assert "genome_map" in info["dataset_types"]
        assert "metadata" in info["dataset_types"]
        assert info["total_files"] == 5
        assert info["last_modified"] == "2023-12-01T10:30:00Z"
        assert info["has_default_config"] is True

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_repository_info_fetch_error(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test getting repository info when structure fetch fails."""
        mock_card_fetcher_instance = Mock()
        mock_structure_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_structure_fetcher.return_value = mock_structure_fetcher_instance

        mock_card_fetcher_instance.fetch.return_value = sample_dataset_card_data
        mock_structure_fetcher_instance.fetch.side_effect = HfDataFetchError(
            "Structure fetch failed"
        )

        datacard = DataCard(test_repo_id)

        info = datacard.get_repository_info()

        assert info["repo_id"] == test_repo_id
        assert info["total_files"] is None
        assert info["last_modified"] is None

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_explore_config(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test exploring a specific configuration."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        # Test regular config
        config_info = datacard.explore_config("binding_data")

        assert config_info["config_name"] == "binding_data"
        assert config_info["description"] == "Transcription factor binding measurements"
        assert config_info["dataset_type"] == "annotated_features"
        assert config_info["is_default"] is False
        assert config_info["num_features"] == 4
        assert len(config_info["features"]) == 4
        assert config_info["metadata_fields"] == [
            "regulator_symbol",
            "experimental_condition",
        ]

        # Test config with partitioning
        partitioned_config_info = datacard.explore_config("genome_map_data")
        assert "partitioning" in partitioned_config_info
        assert partitioned_config_info["partitioning"]["enabled"] is True
        assert partitioned_config_info["partitioning"]["partition_by"] == [
            "regulator",
            "experiment",
        ]

        # Test metadata config with applies_to
        metadata_config_info = datacard.explore_config("experiment_metadata")
        assert metadata_config_info["applies_to"] == ["binding_data"]

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_explore_config_not_found(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test exploring a non-existent configuration."""
        mock_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        with pytest.raises(
            DataCardError, match="Configuration 'nonexistent' not found"
        ):
            datacard.explore_config("nonexistent")

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_summary(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
        sample_repo_structure,
    ):
        """Test getting a summary of the dataset."""
        mock_card_fetcher_instance = Mock()
        mock_structure_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_structure_fetcher.return_value = mock_structure_fetcher_instance

        mock_card_fetcher_instance.fetch.return_value = sample_dataset_card_data
        mock_structure_fetcher_instance.fetch.return_value = sample_repo_structure

        datacard = DataCard(test_repo_id)

        summary = datacard.summary()

        assert "Dataset: Test Genomics Dataset" in summary
        assert f"Repository: {test_repo_id}" in summary
        assert "License: mit" in summary
        assert "Configurations: 4" in summary
        assert "genomic_features" in summary
        assert "binding_data" in summary
        assert "genome_map_data" in summary
        assert "experiment_metadata" in summary
        assert "(default)" in summary  # genomic_features is marked as default

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_extract_partition_values(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test extracting partition values."""
        mock_card_fetcher_instance = Mock()
        mock_structure_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_structure_fetcher.return_value = mock_structure_fetcher_instance

        mock_card_fetcher_instance.fetch.return_value = sample_dataset_card_data
        mock_structure_fetcher_instance.get_partition_values.return_value = [
            "TF1",
            "TF2",
            "TF3",
        ]

        datacard = DataCard(test_repo_id)

        # Get the genome_map_data config which has partitioning enabled
        config = datacard.get_config("genome_map_data")
        assert config is not None
        assert config.dataset_info.partitioning.enabled is True

        values = datacard._extract_partition_values(config, "regulator")
        assert values == {"TF1", "TF2", "TF3"}
        mock_structure_fetcher_instance.get_partition_values.assert_called_once_with(
            test_repo_id, "regulator"
        )

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_extract_partition_values_no_partitioning(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test extracting partition values when partitioning is disabled."""
        mock_card_fetcher_instance = Mock()
        mock_structure_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_structure_fetcher.return_value = mock_structure_fetcher_instance

        mock_card_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        # Get a config without partitioning
        config = datacard.get_config("genomic_features")
        assert config is not None
        assert config.dataset_info.partitioning is None

        values = datacard._extract_partition_values(config, "some_field")
        assert values == set()
        mock_structure_fetcher_instance.get_partition_values.assert_not_called()

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_extract_partition_values_field_not_in_partitions(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test extracting partition values when field is not a partition column."""
        mock_card_fetcher_instance = Mock()
        mock_structure_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_structure_fetcher.return_value = mock_structure_fetcher_instance

        mock_card_fetcher_instance.fetch.return_value = sample_dataset_card_data

        datacard = DataCard(test_repo_id)

        # Get the genome_map_data config which has partitioning enabled
        config = datacard.get_config("genome_map_data")
        assert config is not None

        # Try to extract values for a field that's not in partition_by
        values = datacard._extract_partition_values(config, "not_a_partition_field")
        assert values == set()
        mock_structure_fetcher_instance.get_partition_values.assert_not_called()

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_extract_partition_values_fetch_error(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        sample_dataset_card_data,
    ):
        """Test extracting partition values when fetch fails."""
        mock_card_fetcher_instance = Mock()
        mock_structure_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_structure_fetcher.return_value = mock_structure_fetcher_instance

        mock_card_fetcher_instance.fetch.return_value = sample_dataset_card_data
        mock_structure_fetcher_instance.get_partition_values.side_effect = (
            HfDataFetchError("Fetch failed")
        )

        datacard = DataCard(test_repo_id)

        config = datacard.get_config("genome_map_data")
        values = datacard._extract_partition_values(config, "regulator")

        # Should return empty set on error
        assert values == set()

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_field_attribute(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
    ):
        """Test extracting specific attributes from field definitions."""
        # Create sample card data with condition definitions
        card_data = {
            "configs": [
                {
                    "config_name": "test_config",
                    "description": "Test configuration",
                    "dataset_type": "annotated_features",
                    "data_files": [{"split": "train", "path": "test.parquet"}],
                    "dataset_info": {
                        "features": [
                            {
                                "name": "condition",
                                "dtype": "string",
                                "description": "Experimental condition",
                                "role": "experimental_condition",
                                "definitions": {
                                    "YPD": {
                                        "media": {
                                            "name": "YPD",
                                            "carbon_source": [
                                                {
                                                    "compound": "D-glucose",
                                                    "concentration_percent": 2,
                                                }
                                            ],
                                            "nitrogen_source": [
                                                {
                                                    "compound": "yeast_extract",
                                                    "concentration_percent": 1,
                                                },
                                                {
                                                    "compound": "peptone",
                                                    "concentration_percent": 2,
                                                },
                                            ],
                                        },
                                        "temperature_celsius": 30,
                                    },
                                    "HEAT": {
                                        "media": {
                                            "name": "YPD",
                                            "carbon_source": [
                                                {
                                                    "compound": "D-glucose",
                                                    "concentration_percent": 2,
                                                }
                                            ],
                                        },
                                        "temperature_celsius": 37,
                                    },
                                    "SM": {
                                        "media": {
                                            "name": "synthetic_complete",
                                            "carbon_source": "unspecified",
                                            "nitrogen_source": "unspecified",
                                        }
                                    },
                                },
                            }
                        ]
                    },
                }
            ]
        }

        mock_card_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_card_fetcher_instance.fetch.return_value = card_data

        datacard = DataCard(test_repo_id)

        # Test extracting media attribute
        media_specs = datacard.get_field_attribute(
            "test_config", "condition", "media"
        )

        assert "YPD" in media_specs
        assert "HEAT" in media_specs
        assert "SM" in media_specs

        # Check YPD media specification
        assert media_specs["YPD"]["name"] == "YPD"
        assert len(media_specs["YPD"]["carbon_source"]) == 1
        assert media_specs["YPD"]["carbon_source"][0]["compound"] == "D-glucose"
        assert len(media_specs["YPD"]["nitrogen_source"]) == 2

        # Check HEAT media specification
        assert media_specs["HEAT"]["name"] == "YPD"
        assert len(media_specs["HEAT"]["carbon_source"]) == 1

        # Check SM media specification
        assert media_specs["SM"]["name"] == "synthetic_complete"
        assert media_specs["SM"]["carbon_source"] == "unspecified"

        # Test extracting temperature attribute
        temp_specs = datacard.get_field_attribute(
            "test_config", "condition", "temperature_celsius"
        )

        assert temp_specs["YPD"] == 30
        assert temp_specs["HEAT"] == 37
        assert temp_specs["SM"] == "unspecified"  # SM doesn't have temperature

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_field_attribute_invalid_config(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        minimal_dataset_card_data,
    ):
        """Test get_field_attribute with invalid config name."""
        mock_card_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_card_fetcher_instance.fetch.return_value = minimal_dataset_card_data

        datacard = DataCard(test_repo_id)

        with pytest.raises(DataCardError, match="Configuration 'invalid' not found"):
            datacard.get_field_attribute("invalid", "condition", "media")

    @patch("tfbpapi.datainfo.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datainfo.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datainfo.datacard.HfSizeInfoFetcher")
    def test_get_field_attribute_invalid_field(
        self,
        mock_size_fetcher,
        mock_structure_fetcher,
        mock_card_fetcher,
        test_repo_id,
        minimal_dataset_card_data,
    ):
        """Test get_field_attribute with invalid field name."""
        mock_card_fetcher_instance = Mock()
        mock_card_fetcher.return_value = mock_card_fetcher_instance
        mock_card_fetcher_instance.fetch.return_value = minimal_dataset_card_data

        datacard = DataCard(test_repo_id)

        with pytest.raises(
            DataCardError, match="Field 'invalid_field' not found in config"
        ):
            datacard.get_field_attribute("test_config", "invalid_field", "media")
