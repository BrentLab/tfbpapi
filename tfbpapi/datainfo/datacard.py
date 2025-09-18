"""DataCard class for easy exploration of HuggingFace dataset metadata."""

import logging
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import ValidationError

from ..errors import (
    DataCardError,
    DataCardValidationError,
    HfDataFetchError,
)
from .fetchers import HfDataCardFetcher, HfRepoStructureFetcher, HfSizeInfoFetcher
from .models import (
    DatasetCard,
    DatasetConfig,
    DatasetType,
    ExtractedMetadata,
    MetadataRelationship,
)


class DataCard:
    """
    Easy-to-use interface for exploring HuggingFace dataset metadata.

    Provides methods to discover and explore dataset contents, configurations, and
    metadata without loading the actual genomic data.

    """

    def __init__(self, repo_id: str, token: str | None = None):
        """
        Initialize DataCard for a repository.

        :param repo_id: HuggingFace repository identifier (e.g., "user/dataset")
        :param token: Optional HuggingFace token for authentication

        """
        self.repo_id = repo_id
        self.token = token
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize fetchers
        self._card_fetcher = HfDataCardFetcher(token=token)
        self._structure_fetcher = HfRepoStructureFetcher(token=token)
        self._size_fetcher = HfSizeInfoFetcher(token=token)

        # Cache for parsed card
        self._dataset_card: DatasetCard | None = None
        self._metadata_cache: dict[str, list[ExtractedMetadata]] = {}

    @property
    def dataset_card(self) -> DatasetCard:
        """Get the validated dataset card."""
        if self._dataset_card is None:
            self._load_and_validate_card()
        # this is here for type checking purposes. _load_and_validate_card()
        # will either set the _dataset_card or raise an error
        assert self._dataset_card is not None
        return self._dataset_card

    def _load_and_validate_card(self) -> None:
        """Load and validate the dataset card from HuggingFace."""
        try:
            self.logger.debug(f"Loading dataset card for {self.repo_id}")
            card_data = self._card_fetcher.fetch(self.repo_id)

            if not card_data:
                raise DataCardValidationError(
                    f"No dataset card found for {self.repo_id}"
                )

            # Validate using Pydantic model
            self._dataset_card = DatasetCard(**card_data)
            self.logger.debug(f"Successfully validated dataset card for {self.repo_id}")

        except ValidationError as e:
            # Create a more user-friendly error message
            error_details = []
            for error in e.errors():
                field_path = " -> ".join(str(x) for x in error["loc"])
                error_type = error["type"]
                error_msg = error["msg"]
                input_value = error.get("input", "N/A")

                if "dtype" in field_path and error_type == "string_type":
                    error_details.append(
                        f"Field '{field_path}': Expected a simple data type string (like 'string', 'int64', 'float64') "
                        f"but got a complex structure. This might be a categorical field with class labels. "
                        f"Actual value: {input_value}"
                    )
                else:
                    error_details.append(
                        f"Field '{field_path}': {error_msg} (got: {input_value})"
                    )

            detailed_msg = (
                f"Dataset card validation failed for {self.repo_id}:\n"
                + "\n".join(f"  - {detail}" for detail in error_details)
            )
            self.logger.error(detailed_msg)
            raise DataCardValidationError(detailed_msg) from e
        except HfDataFetchError as e:
            raise DataCardError(f"Failed to fetch dataset card: {e}") from e

    @property
    def configs(self) -> list[DatasetConfig]:
        """Get all dataset configurations."""
        return self.dataset_card.configs

    def get_config(self, config_name: str) -> DatasetConfig | None:
        """Get a specific configuration by name."""
        return self.dataset_card.get_config_by_name(config_name)

    def get_configs_by_type(
        self, dataset_type: DatasetType | str
    ) -> list[DatasetConfig]:
        """Get configurations by dataset type."""
        if isinstance(dataset_type, str):
            dataset_type = DatasetType(dataset_type)
        return self.dataset_card.get_configs_by_type(dataset_type)

    def get_regulators(self, config_name: str | None = None) -> set[str]:
        """
        Get all regulators mentioned in the dataset.

        :param config_name: Optional specific config to search, otherwise searches all
        :return: Set of regulator identifiers found

        """
        raise NotImplementedError("Method not yet implemented")

    def get_experimental_conditions(self, config_name: str | None = None) -> set[str]:
        """
        Get all experimental conditions mentioned in the dataset.

        :param config_name: Optional specific config to search, otherwise searches all
        :return: Set of experimental conditions found

        """
        raise NotImplementedError("Method not yet implemented")

    def get_field_values(self, config_name: str, field_name: str) -> set[str]:
        """
        Get all unique values for a specific field in a configuration.

        :param config_name: Configuration name
        :param field_name: Field name to extract values from
        :return: Set of unique values
        :raises DataCardError: If config or field not found

        """
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        # Check if field exists in the config
        field_names = [f.name for f in config.dataset_info.features]
        if field_name not in field_names:
            raise DataCardError(
                f"Field '{field_name}' not found in config '{config_name}'"
            )

        return self._extract_field_values(config, field_name)

    def _extract_field_values(self, config: DatasetConfig, field_name: str) -> set[str]:
        """Extract unique values for a field from various sources."""
        values = set()

        # Check cache first
        cache_key = f"{config.config_name}:{field_name}"
        if cache_key in self._metadata_cache:
            cached_metadata = self._metadata_cache[cache_key]
            for meta in cached_metadata:
                if meta.field_name == field_name:
                    values.update(meta.values)
                    return values

        try:
            # For partitioned datasets, extract from file structure
            if (
                config.dataset_info.partitioning
                and config.dataset_info.partitioning.enabled
            ):
                partition_values = self._extract_partition_values(config, field_name)
                if partition_values:
                    values.update(partition_values)

            # For embedded metadata fields, we would need to query the actual data
            # This is a placeholder - in practice, you might use the HF datasets server API
            if config.metadata_fields and field_name in config.metadata_fields:
                # Placeholder for actual data extraction
                self.logger.debug(
                    f"Would extract embedded metadata for {field_name} in {config.config_name}"
                )

        except Exception as e:
            self.logger.warning(f"Failed to extract values for {field_name}: {e}")

        return values

    def _extract_partition_values(
        self, config: DatasetConfig, field_name: str
    ) -> set[str]:
        """Extract values from partition structure."""
        if (
            not config.dataset_info.partitioning
            or not config.dataset_info.partitioning.enabled
        ):
            return set()

        partition_columns = config.dataset_info.partitioning.partition_by or []
        if field_name not in partition_columns:
            return set()

        try:
            # Get partition values from repository structure
            partition_values = self._structure_fetcher.get_partition_values(
                self.repo_id, field_name
            )
            return set(partition_values)
        except HfDataFetchError:
            self.logger.warning(f"Failed to extract partition values for {field_name}")
            return set()

    def get_metadata_relationships(self) -> list[MetadataRelationship]:
        """Get relationships between data configs and their metadata."""
        relationships = []
        data_configs = self.dataset_card.get_data_configs()
        metadata_configs = self.dataset_card.get_metadata_configs()

        for data_config in data_configs:
            # Check for explicit applies_to relationships
            for meta_config in metadata_configs:
                if (
                    meta_config.applies_to
                    and data_config.config_name in meta_config.applies_to
                ):
                    relationships.append(
                        MetadataRelationship(
                            data_config=data_config.config_name,
                            metadata_config=meta_config.config_name,
                            relationship_type="explicit",
                        )
                    )
                    continue

            # Check for embedded metadata
            if data_config.metadata_fields:
                relationships.append(
                    MetadataRelationship(
                        data_config=data_config.config_name,
                        metadata_config=f"{data_config.config_name}_embedded",
                        relationship_type="embedded",
                    )
                )

        return relationships

    def get_repository_info(self) -> dict[str, Any]:
        """Get general repository information."""
        card = self.dataset_card

        try:
            structure = self._structure_fetcher.fetch(self.repo_id)
            total_files = structure.get("total_files", 0)
            last_modified = structure.get("last_modified")
        except HfDataFetchError:
            total_files = None
            last_modified = None

        return {
            "repo_id": self.repo_id,
            "pretty_name": card.pretty_name,
            "license": card.license,
            "tags": card.tags,
            "language": card.language,
            "size_categories": card.size_categories,
            "num_configs": len(card.configs),
            "dataset_types": [config.dataset_type.value for config in card.configs],
            "total_files": total_files,
            "last_modified": last_modified,
            "has_default_config": self.dataset_card.get_default_config() is not None,
        }

    def explore_config(self, config_name: str) -> dict[str, Any]:
        """Get detailed information about a specific configuration."""
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        info: dict[str, Any] = {
            "config_name": config.config_name,
            "description": config.description,
            "dataset_type": config.dataset_type.value,
            "is_default": config.default,
            "num_features": len(config.dataset_info.features),
            "features": [
                {"name": f.name, "dtype": f.dtype, "description": f.description}
                for f in config.dataset_info.features
            ],
            "data_files": [
                {"split": df.split, "path": df.path} for df in config.data_files
            ],
        }

        # Add partitioning info if present
        if config.dataset_info.partitioning:
            info["partitioning"] = {
                "enabled": config.dataset_info.partitioning.enabled,
                "partition_by": config.dataset_info.partitioning.partition_by,
                "path_template": config.dataset_info.partitioning.path_template,
            }

        # Add metadata-specific fields
        if config.applies_to:
            info["applies_to"] = config.applies_to

        if config.metadata_fields:
            info["metadata_fields"] = config.metadata_fields

        return info

    def summary(self) -> str:
        """Get a human-readable summary of the dataset."""
        card = self.dataset_card
        info = self.get_repository_info()

        lines = [
            f"Dataset: {card.pretty_name or self.repo_id}",
            f"Repository: {self.repo_id}",
            f"License: {card.license or 'Not specified'}",
            f"Configurations: {len(card.configs)}",
            f"Dataset Types: {', '.join(info['dataset_types'])}",
        ]

        if card.tags:
            lines.append(f"Tags: {', '.join(card.tags)}")

        # Add config summaries
        lines.append("\nConfigurations:")
        for config in card.configs:
            default_mark = " (default)" if config.default else ""
            lines.append(
                f"  - {config.config_name}: {config.dataset_type.value}{default_mark}"
            )
            lines.append(f"    {config.description}")

        return "\n".join(lines)
