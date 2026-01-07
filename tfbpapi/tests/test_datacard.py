"""Tests for the DataCard class."""

from unittest.mock import Mock, patch

import pytest

from tfbpapi import DataCard
from tfbpapi.errors import DataCardError, DataCardValidationError, HfDataFetchError
from tfbpapi.models import DatasetType


class TestDataCard:
    """Test suite for DataCard class."""

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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
        assert config.dataset_info.partitioning.enabled is True  # type: ignore

        values = datacard._extract_partition_values(config, "regulator")
        assert values == {"TF1", "TF2", "TF3"}
        mock_structure_fetcher_instance.get_partition_values.assert_called_once_with(
            test_repo_id, "regulator"
        )

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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

    @patch("tfbpapi.datacard.HfDataCardFetcher")
    @patch("tfbpapi.datacard.HfRepoStructureFetcher")
    @patch("tfbpapi.datacard.HfSizeInfoFetcher")
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
        values = datacard._extract_partition_values(config, "regulator")  # type: ignore

        # Should return empty set on error
        assert values == set()
