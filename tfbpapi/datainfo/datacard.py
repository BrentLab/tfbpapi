"""
DataCard class for parsing and exploring HuggingFace dataset metadata.

This module provides the DataCard class for parsing HuggingFace dataset cards
into structured Python objects that can be easily explored. The focus is on
enabling users to drill down into the YAML structure to understand:

- Dataset configurations and their types
- Feature definitions and roles
- Experimental conditions at all hierarchy levels (top/config/field)
- Field-level condition definitions
- Metadata relationships

Users can then use this information to plan metadata table structures and
data loading strategies.

"""

import logging
from typing import Any

from pydantic import ValidationError

from ..errors import DataCardError, DataCardValidationError, HfDataFetchError
from .fetchers import HfDataCardFetcher, HfRepoStructureFetcher, HfSizeInfoFetcher
from .models import (
    DatasetCard,
    DatasetConfig,
    DatasetType,
    ExtractedMetadata,
    FeatureInfo,
    MetadataRelationship,
)


class DataCard:
    """
    Parser and explorer for HuggingFace dataset metadata.

    DataCard parses HuggingFace dataset cards into flexible Python objects,
    enabling users to drill down into the YAML structure to understand dataset
    organization, experimental conditions, and metadata relationships.

    The parsed structure uses Pydantic models with `extra="allow"` to accept
    arbitrary fields (like experimental_conditions) without requiring code
    changes. This makes the system flexible enough to handle domain-specific
    metadata variations.

    Key capabilities:
    - Parse dataset card YAML into structured objects
    - Navigate experimental conditions at 3 levels (top/config/field)
    - Explore field definitions and roles
    - Extract metadata schema for table design
    - Discover metadata relationships

    Example (new API):
        >>> card = DataCard("BrentLab/harbison_2004")
        >>> # Use context manager for config exploration
        >>> with card.config("harbison_2004") as cfg:
        ...     # Get all experimental conditions
        ...     conds = cfg.experimental_conditions()
        ...     # Get condition fields with definitions
        ...     fields = cfg.condition_fields()
        ...     # Drill down into specific field
        ...     for name, info in fields.items():
        ...         for value, definition in info['definitions'].items():
        ...             print(f"{name}={value}: {definition}")

    Example (legacy API still supported):
        >>> card = DataCard("BrentLab/harbison_2004")
        >>> conditions = card.get_experimental_conditions("harbison_2004")
        >>> defs = card.get_field_definitions("harbison_2004", "condition")

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
                        f"Field '{field_path}': Expected a simple data type "
                        "string (like 'string', 'int64', 'float64') "
                        "but got a complex structure. This might be a categorical "
                        "field with class labels. "
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

    def get_card_metadata(self) -> dict[str, Any]:
        """
        Get all top-level metadata fields from the dataset card.

        Returns all fields stored in model_extra (e.g., license, tags, pretty_name,
        etc.) as a dict. This gives direct access to the raw YAML structure at the card
        level.

        :return: Dict of all extra metadata fields

        """
        if self.dataset_card.model_extra:
            return dict(self.dataset_card.model_extra)
        return {}

    def get_config_metadata(self, config_name: str) -> dict[str, Any]:
        """
        Get all extra metadata fields from a specific config.

        Returns all fields stored in the config's model_extra (e.g.,
        experimental_conditions, custom fields, etc.) as a dict.

        :param config_name: Configuration name
        :return: Dict of all extra metadata fields for this config
        :raises DataCardError: If config not found

        """
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        if config.model_extra:
            return dict(config.model_extra)
        return {}

    def get_features(self, config_name: str) -> list[FeatureInfo]:
        """
        Get all feature definitions for a configuration.

        :param config_name: Configuration name
        :return: List of FeatureInfo objects
        :raises DataCardError: If config not found

        """
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        return config.dataset_info.features

    def get_features_by_role(
        self, config_name: str, role: str | None = None
    ) -> dict[str, list[str]]:
        """
        Get features grouped by role.

        If role is specified, returns only features with that role.
        If role is None, returns all features grouped by role.

        :param config_name: Configuration name
        :param role: Optional specific role to filter by
        :return: Dict mapping role -> list of field names
        :raises DataCardError: If config not found

        Example:
            >>> # Get all features by role
            >>> by_role = card.get_features_by_role("config_name")
            >>> # {'regulator_identifier': ['regulator_locus_tag'],
            >>> #  'target_identifier': ['target_locus_tag'], ...}
            >>>
            >>> # Get only experimental condition features
            >>> cond_fields = card.get_features_by_role("config_name",
            ...                                          "experimental_condition")
            >>> # {'experimental_condition': ['condition', 'treatment']}

        """
        features = self.get_features(config_name)

        # Group by role
        by_role: dict[str, list[str]] = {}
        for feature in features:
            feature_role = feature.role if feature.role else "no_role"
            if feature_role not in by_role:
                by_role[feature_role] = []
            by_role[feature_role].append(feature.name)

        # Filter by specific role if requested
        if role is not None:
            return {role: by_role.get(role, [])}

        return by_role

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
        """Extract unique values for a field from partition structure only."""
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
                    # Cache the result
                    self._metadata_cache[cache_key] = [
                        ExtractedMetadata(
                            config_name=config.config_name,
                            field_name=field_name,
                            values=values,
                            extraction_method="partition_structure",
                        )
                    ]
                    return values

            # For non-partitioned fields, we can no longer query parquet files
            self.logger.debug(
                f"Cannot extract values for {field_name} in {config.config_name}: "
                "field is not partitioned and parquet querying is not supported"
            )

        except Exception as e:
            self.logger.warning(f"Failed to extract values for {field_name}: {e}")
            # Return empty set on failure instead of raising
            # This maintains backward compatibility

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

    def get_metadata_relationships(
        self, refresh_cache: bool = False
    ) -> list[MetadataRelationship]:
        """
        Get relationships between data configs and their metadata.

        :param refresh_cache: If True, force refresh dataset card from remote

        """
        # Clear cached dataset card if refresh requested
        if refresh_cache:
            self._dataset_card = None

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

            # Check for embedded metadata (always runs regardless of
            # explicit relationships)
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

    def explore_config(
        self, config_name: str, include_extra: bool = True
    ) -> dict[str, Any]:
        """
        Get detailed information about a specific configuration.

        Returns a comprehensive dict with config structure including features, data
        files, partitioning, and optionally all extra metadata fields.

        :param config_name: Configuration name
        :param include_extra: If True, include all fields from model_extra
        :return: Dict with config details
        :raises DataCardError: If config not found

        """
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
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "description": f.description,
                    "role": f.role,
                    "has_definitions": f.definitions is not None,
                }
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

        # Add all extra fields from model_extra
        if include_extra and config.model_extra:
            info["extra_fields"] = dict(config.model_extra)

        return info

    def extract_metadata_schema(self, config_name: str) -> dict[str, Any]:
        """
        Extract complete metadata schema for planning metadata table structure.

        This is the primary method for understanding what metadata is available and
        how to structure it into a metadata table. It consolidates information from
        all sources:

        - **Field roles**: Which fields are regulators, targets, conditions, etc.
        - **Top-level conditions**: Repo-wide conditions (constant for all samples)
        - **Config-level conditions**: Config-specific conditions (constant for this config)
        - **Field-level definitions**: Per-sample condition definitions

        The returned schema provides all the information needed to:
        1. Identify sample identifier fields (regulator_identifier, etc.)
        2. Determine which conditions are constant vs. variable
        3. Access condition definitions for creating flattened columns
        4. Plan metadata table structure

        :param config_name: Configuration name to extract schema for
        :return: Dict with comprehensive schema including:
            - regulator_fields: List of regulator identifier field names
            - target_fields: List of target identifier field names
            - condition_fields: List of experimental_condition field names
            - condition_definitions: Dict mapping field -> value -> definition
            - top_level_conditions: Dict of repo-wide conditions
            - config_level_conditions: Dict of config-specific conditions
        :raises DataCardError: If configuration not found

        Example:
            >>> schema = card.extract_metadata_schema('harbison_2004')
            >>> # Identify identifier fields
            >>> print(f"Regulator fields: {schema['regulator_fields']}")
            >>> # Check for constant conditions
            >>> if schema['top_level_conditions']:
            ...     print("Has repo-wide constant conditions")
            >>> # Get field-level definitions for metadata table
            >>> for field in schema['condition_fields']:
            ...     defs = schema['condition_definitions'][field]
            ...     print(f"{field} has {len(defs)} levels")

        """
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        schema: dict[str, Any] = {
            "regulator_fields": [],  # Fields with role=regulator_identifier
            "target_fields": [],  # Fields with role=target_identifier
            "condition_fields": [],  # Fields with role=experimental_condition
            "condition_definitions": {},  # Field-level condition details
            "top_level_conditions": None,  # Repo-level conditions
            "config_level_conditions": None,  # Config-level conditions
        }

        for feature in config.dataset_info.features:
            if feature.role == "regulator_identifier":
                schema["regulator_fields"].append(feature.name)
            elif feature.role == "target_identifier":
                schema["target_fields"].append(feature.name)
            elif feature.role == "experimental_condition":
                schema["condition_fields"].append(feature.name)
                if feature.definitions:
                    schema["condition_definitions"][feature.name] = feature.definitions

        # Add top-level conditions (applies to all configs/samples)
        # Stored in model_extra as dict
        if self.dataset_card.model_extra:
            top_level = self.dataset_card.model_extra.get("experimental_conditions")
            if top_level:
                schema["top_level_conditions"] = top_level

        # Add config-level conditions (applies to this config's samples)
        # Stored in model_extra as dict
        if config.model_extra:
            config_level = config.model_extra.get("experimental_conditions")
            if config_level:
                schema["config_level_conditions"] = config_level

        return schema

    def get_condition_levels(
        self, config_name: str, field_name: str
    ) -> dict[str, Any] | list[str]:
        """
        Get factor levels for an experimental condition field.

        Returns definitions if available (structured dict with descriptions), otherwise
        queries distinct values from the parquet file.

        :param config_name: Configuration name
        :param field_name: Experimental condition field name
        :return: Dict of definitions if available, otherwise list of distinct values
        :raises DataCardError: If config or field not found, or field is not an
            experimental condition

        """
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        # Find the feature and verify it's an experimental condition
        feature = None
        for f in config.dataset_info.features:
            if f.name == field_name:
                feature = f
                break

        if not feature:
            raise DataCardError(
                f"Field '{field_name}' not found in config '{config_name}'"
            )

        if feature.role != "experimental_condition":
            raise DataCardError(
                f"Field '{field_name}' is not an experimental condition "
                f"(role={feature.role})"
            )

        # If field has definitions, return those
        if feature.definitions:
            return feature.definitions

        # Otherwise, query distinct values from parquet file
        values = self.get_field_values(config_name, field_name)
        return sorted(list(values))

    def get_experimental_conditions(
        self, config_name: str | None = None
    ) -> dict[str, Any]:
        """
        Get experimental conditions with proper hierarchy handling.

        This method enables drilling down into the experimental conditions hierarchy:
        - Top-level (repo-wide): Common to all configs/samples
        - Config-level: Specific to a config, common to its samples
        - Field-level: Per-sample variation (use get_field_definitions instead)

        Returns experimental conditions at the appropriate level:
        - If config_name is None: returns top-level (repo-wide) conditions only
        - If config_name is provided: returns merged (top + config) conditions

        All conditions are returned as flexible dicts that preserve the original
        YAML structure. Navigate nested dicts to access specific values.

        :param config_name: Optional config name. If provided, merges top and config levels
        :return: Dict of experimental conditions (empty dict if none defined)

        Example:
            >>> # Get top-level conditions
            >>> top = card.get_experimental_conditions()
            >>> temp = top.get('temperature_celsius', 30)
            >>>
            >>> # Get merged conditions for a config
            >>> merged = card.get_experimental_conditions('config_name')
            >>> media = merged.get('media', {})
            >>> media_name = media.get('name', 'unspecified')

        """
        # Get top-level conditions (stored in model_extra)
        top_level = (
            self.dataset_card.model_extra.get("experimental_conditions", {})
            if self.dataset_card.model_extra
            else {}
        )

        # If no config specified, return top-level only
        if config_name is None:
            return top_level.copy() if isinstance(top_level, dict) else {}

        # Get config-level conditions
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        config_level = (
            config.model_extra.get("experimental_conditions", {})
            if config.model_extra
            else {}
        )

        # Merge: config-level overrides top-level
        merged = {}
        if isinstance(top_level, dict):
            merged.update(top_level)
        if isinstance(config_level, dict):
            merged.update(config_level)

        return merged

    def get_field_definitions(
        self, config_name: str, field_name: str
    ) -> dict[str, Any]:
        """
        Get definitions for a specific field (field-level conditions).

        This is the third level of the experimental conditions hierarchy - conditions
        that vary per sample. Returns a dict mapping each possible field value to its
        detailed specification.

        For fields with role=experimental_condition, the definitions typically include
        nested structures like media composition, temperature, treatments, etc. that
        define what each categorical value means experimentally.

        :param config_name: Configuration name
        :param field_name: Field name (typically has role=experimental_condition)
        :return: Dict mapping field values to their definition dicts (empty if no definitions)
        :raises DataCardError: If config or field not found

        Example:
            >>> # Get condition definitions
            >>> defs = card.get_field_definitions('harbison_2004', 'condition')
            >>> # defs = {'YPD': {...}, 'HEAT': {...}, ...}
            >>>
            >>> # Drill down into a specific condition
            >>> ypd = defs['YPD']
            >>> env_conds = ypd.get('environmental_conditions', {})
            >>> media = env_conds.get('media', {})
            >>> media_name = media.get('name')

        """
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        # Find the feature
        feature = None
        for f in config.dataset_info.features:
            if f.name == field_name:
                feature = f
                break

        if not feature:
            raise DataCardError(
                f"Field '{field_name}' not found in config '{config_name}'"
            )

        # Return definitions if present, otherwise empty dict
        return feature.definitions if feature.definitions else {}

    def get_field_attribute(
        self, config_name: str, field_name: str, attribute: str
    ) -> dict[str, Any]:
        """
        Extract a specific attribute from field definitions.

        This is useful for exploring nested attributes in condition definitions,
        such as media composition, temperature parameters, or growth phases.

        :param config_name: Configuration name
        :param field_name: Field with definitions (e.g., 'condition')
        :param attribute: Attribute to extract (e.g., 'media', 'temperature_celsius')
        :return: Dict mapping field values to their attribute specifications.
            Returns 'unspecified' if attribute doesn't exist for a value.

        Example:
            >>> card = DataCard('BrentLab/harbison_2004')
            >>> media = card.get_field_attribute('harbison_2004', 'condition', 'media')
            >>> print(media['YPD'])
            {'name': 'YPD', 'carbon_source': [...], 'nitrogen_source': [...]}

        """
        # Get all field definitions
        definitions = self.get_field_definitions(config_name, field_name)

        # Extract attribute for each definition
        result = {}
        for field_value, definition in definitions.items():
            if attribute in definition:
                result[field_value] = definition[attribute]
            else:
                result[field_value] = "unspecified"

        return result

    def list_experimental_condition_fields(self, config_name: str) -> list[str]:
        """
        List all fields with role=experimental_condition in a config.

        These are fields that contain per-sample experimental condition variation.
        They represent the field-level (third level) of the experimental conditions
        hierarchy.

        Fields with this role typically have `definitions` that map each value to
        its detailed experimental specification. Use `get_field_definitions()` to
        access these definitions.

        :param config_name: Configuration name
        :return: List of field names with experimental_condition role
        :raises DataCardError: If config not found

        Example:
            >>> # Find all condition fields
            >>> cond_fields = card.list_experimental_condition_fields('config_name')
            >>> # ['condition', 'treatment', 'time_point']
            >>>
            >>> # Then get definitions for each
            >>> for field in cond_fields:
            ...     defs = card.get_field_definitions('config_name', field)
            ...     print(f"{field}: {len(defs)} levels")

        """
        config = self.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        return [
            f.name
            for f in config.dataset_info.features
            if f.role == "experimental_condition"
        ]

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
