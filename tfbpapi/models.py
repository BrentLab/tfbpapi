"""
Pydantic models for dataset card validation and metadata configuration.

These models provide minimal structure for parsing HuggingFace dataset cards while
remaining flexible enough to accommodate diverse experimental systems. Most fields use
extra="allow" to accept domain-specific additions without requiring code changes.

Also includes models for VirtualDB metadata normalization configuration.

"""

from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetType(str, Enum):
    """Supported dataset types."""

    GENOMIC_FEATURES = "genomic_features"
    ANNOTATED_FEATURES = "annotated_features"
    GENOME_MAP = "genome_map"
    METADATA = "metadata"
    QC_DATA = "qc_data"


class FeatureInfo(BaseModel):
    """
    Information about a dataset feature/column.

    Minimal required fields with flexible dtype handling.

    """

    name: str = Field(..., description="Column name in the data")
    dtype: str | dict[str, Any] = Field(
        ...,
        description="Data type (string, int64, float64, etc.) or class_label dict",
    )
    description: str = Field(..., description="Description of the field")
    role: str | None = Field(
        default=None,
        description="Optional semantic role. 'experimental_condition' has special behavior.",
    )
    definitions: dict[str, Any] | None = Field(
        default=None,
        description="For experimental_condition fields: definitions per value",
    )


class PartitioningInfo(BaseModel):
    """Partitioning configuration for datasets."""

    enabled: bool = Field(default=False, description="Whether partitioning is enabled")
    partition_by: list[str] | None = Field(
        default=None, description="Partition column names"
    )
    path_template: str | None = Field(
        default=None, description="Path template for partitioned files"
    )


class DatasetInfo(BaseModel):
    """Dataset structure information."""

    features: list[FeatureInfo] = Field(..., description="Feature definitions")
    partitioning: PartitioningInfo | None = Field(
        default=None, description="Partitioning configuration"
    )


class DataFileInfo(BaseModel):
    """Information about data files."""

    split: str = Field(default="train", description="Dataset split name")
    path: str = Field(..., description="Path to data file(s)")


class DatasetConfig(BaseModel):
    """
    Configuration for a dataset within a repository.

    Uses extra="allow" to accept arbitrary experimental_conditions and other fields.

    """

    config_name: str = Field(..., description="Unique configuration identifier")
    description: str = Field(..., description="Human-readable description")
    dataset_type: DatasetType = Field(..., description="Type of dataset")
    default: bool = Field(
        default=False, description="Whether this is the default config"
    )
    applies_to: list[str] | None = Field(
        default=None, description="Configs this metadata applies to"
    )
    metadata_fields: list[str] | None = Field(
        default=None, description="Fields for embedded metadata extraction"
    )
    data_files: list[DataFileInfo] = Field(..., description="Data file information")
    dataset_info: DatasetInfo = Field(..., description="Dataset structure information")

    model_config = ConfigDict(extra="allow")

    @field_validator("applies_to")
    @classmethod
    def applies_to_only_for_metadata(cls, v, info):
        """Validate that applies_to is only used for metadata or qc_data configs."""
        if v is not None:
            dataset_type = info.data.get("dataset_type")
            if dataset_type not in (DatasetType.METADATA, DatasetType.QC_DATA):
                raise ValueError(
                    "applies_to field is only valid "
                    "for metadata and qc_data dataset types"
                )
        return v

    @field_validator("metadata_fields")
    @classmethod
    def metadata_fields_validation(cls, v):
        """Validate metadata_fields usage."""
        if v is not None and len(v) == 0:
            raise ValueError("metadata_fields cannot be empty list, use None instead")
        return v


class DatasetCard(BaseModel):
    """
    Complete dataset card model.

    Uses extra="allow" to accept arbitrary top-level metadata and
    experimental_conditions.

    """

    configs: list[DatasetConfig] = Field(..., description="Dataset configurations")

    model_config = ConfigDict(extra="allow")

    @field_validator("configs")
    @classmethod
    def configs_not_empty(cls, v):
        """Ensure at least one config is present."""
        if not v:
            raise ValueError("At least one dataset configuration is required")
        return v

    @field_validator("configs")
    @classmethod
    def unique_config_names(cls, v):
        """Ensure config names are unique."""
        names = [config.config_name for config in v]
        if len(names) != len(set(names)):
            raise ValueError("Configuration names must be unique")
        return v

    @field_validator("configs")
    @classmethod
    def at_most_one_default(cls, v):
        """Ensure at most one config is marked as default."""
        defaults = [config for config in v if config.default]
        if len(defaults) > 1:
            raise ValueError("At most one configuration can be marked as default")
        return v

    def get_config_by_name(self, name: str) -> DatasetConfig | None:
        """Get a configuration by name."""
        for config in self.configs:
            if config.config_name == name:
                return config
        return None

    def get_configs_by_type(self, dataset_type: DatasetType) -> list[DatasetConfig]:
        """Get all configurations of a specific type."""
        return [
            config for config in self.configs if config.dataset_type == dataset_type
        ]

    def get_default_config(self) -> DatasetConfig | None:
        """Get the default configuration if one exists."""
        defaults = [config for config in self.configs if config.default]
        return defaults[0] if defaults else None

    def get_data_configs(self) -> list[DatasetConfig]:
        """Get all non-metadata configurations."""
        return [
            config
            for config in self.configs
            if config.dataset_type != DatasetType.METADATA
        ]

    def get_metadata_configs(self) -> list[DatasetConfig]:
        """Get all metadata configurations."""
        return [
            config
            for config in self.configs
            if config.dataset_type == DatasetType.METADATA
        ]


class ExtractedMetadata(BaseModel):
    """Metadata extracted from datasets."""

    config_name: str = Field(..., description="Source configuration name")
    field_name: str = Field(
        ..., description="Field name the metadata was extracted from"
    )
    values: set[str] = Field(..., description="Unique values found")
    extraction_method: str = Field(..., description="How the metadata was extracted")

    model_config = ConfigDict(
        # Allow sets in JSON serialization
        json_encoders={set: list}
    )


class MetadataRelationship(BaseModel):
    """Relationship between a data config and its metadata."""

    data_config: str = Field(..., description="Data configuration name")
    metadata_config: str = Field(..., description="Metadata configuration name")
    relationship_type: str = Field(
        ..., description="Type of relationship (explicit, embedded)"
    )


# ============================================================================
# VirtualDB Metadata Configuration Models
# ============================================================================


class PropertyMapping(BaseModel):
    """
    Mapping specification for a single property.

    Attributes:
        path: Optional dot-notation path to the property value.
              For repo/config-level: relative to experimental_conditions
              For field-level: relative to field definitions
              When omitted with field specified, creates a column alias.
        field: Optional field name for field-level properties.
               When specified, looks in this field's definitions.
               When omitted, looks in repo/config-level experimental_conditions.

    Examples:
        Field-level property with path:
            PropertyMapping(field="condition", path="media.carbon_source")

        Repo/config-level property:
            PropertyMapping(path="temperature_celsius")

        Field-level column alias (no path):
            PropertyMapping(field="condition")

    """

    field: str | None = Field(None, description="Field name for field-level properties")
    path: str | None = Field(None, description="Dot-notation path to property")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str | None) -> str | None:
        """Ensure path is not just whitespace if provided."""
        if v is not None and not v.strip():
            raise ValueError("path cannot be empty or whitespace")
        return v.strip() if v else None

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str | None) -> str | None:
        """Ensure field is not empty string if provided."""
        if v is not None and not v.strip():
            raise ValueError("field cannot be empty or whitespace")
        return v.strip() if v else None

    @model_validator(mode="after")
    def validate_at_least_one_specified(self) -> "PropertyMapping":
        """Ensure at least field or path is specified."""
        if self.field is None and self.path is None:
            raise ValueError("At least one of 'field' or 'path' must be specified")
        return self


class RepositoryConfig(BaseModel):
    """
    Configuration for a single repository. Eg BrentLab/harbison_2004.

    Attributes:
        properties: Repo-wide property mappings that apply to all datasets
        dataset: Dataset-specific property mappings (override repo-wide)

    Example:
        ```python
        config = RepositoryConfig(
            properties={
                "temperature_celsius": PropertyMapping(path="temperature_celsius")
            },
            dataset={
                "dataset_name": {
                    "carbon_source": PropertyMapping(
                        field="condition",
                        path="media.carbon_source"
                    )
                }
            }
        )
        ```

    """

    properties: dict[str, PropertyMapping] = Field(
        default_factory=dict, description="Repo-wide property mappings"
    )
    dataset: dict[str, dict[str, PropertyMapping]] | None = Field(
        None, description="Dataset-specific property mappings"
    )

    @model_validator(mode="before")
    @classmethod
    def parse_structure(cls, data: Any) -> Any:
        """Parse raw dict structure into typed PropertyMapping objects."""
        if not isinstance(data, dict):
            return data

        # Extract and parse dataset section
        dataset_section = data.get("dataset")
        parsed_datasets: dict[str, dict[str, PropertyMapping]] | None = None

        if dataset_section:
            if not isinstance(dataset_section, dict):
                raise ValueError("'dataset' key must contain a dict")

            parsed_datasets = {}
            for dataset_name, properties in dataset_section.items():
                if not isinstance(properties, dict):
                    raise ValueError(
                        f"Dataset '{dataset_name}' must contain a dict of properties"
                    )

                # Parse each property mapping into PropertyMapping object
                parsed_datasets[dataset_name] = {}
                for prop_name, mapping in properties.items():
                    try:
                        parsed_datasets[dataset_name][prop_name] = (
                            PropertyMapping.model_validate(mapping)
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Invalid property '{prop_name}' in dataset "
                            f"'{dataset_name}': {e}"
                        ) from e

        # Parse repo-wide properties (all keys except 'dataset')
        parsed_properties = {}
        for key, value in data.items():
            if key == "dataset":
                continue

            try:
                parsed_properties[key] = PropertyMapping.model_validate(value)
            except Exception as e:
                raise ValueError(f"Invalid repo-wide property '{key}': {e}") from e

        return {"properties": parsed_properties, "dataset": parsed_datasets}


class MetadataConfig(BaseModel):
    """
    Configuration for building standardized metadata tables.

    Specifies optional alias mappings for normalizing factor levels across
    heterogeneous datasets, plus property path mappings for each repository.

    Attributes:
        factor_aliases: Optional mappings of standardized names to actual values.
                        Example: {"carbon_source": {"glucose": ["D-glucose", "dextrose"]}}
        missing_value_labels: Labels for missing values by property name
        description: Human-readable descriptions for each property
        repositories: Dict mapping repository IDs to their configurations

    Example:
        ```yaml
        repositories:
          BrentLab/harbison_2004:
            dataset:
              harbison_2004:
                carbon_source:
                  field: condition
                  path: media.carbon_source

          BrentLab/kemmeren_2014:
            temperature:
              path: temperature_celsius
            dataset:
              kemmeren_2014:
                carbon_source:
                  path: media.carbon_source

        factor_aliases:
          carbon_source:
            glucose: ["D-glucose", "dextrose"]
            galactose: ["D-galactose", "Galactose"]

        missing_value_labels:
          carbon_source: "unspecified"

        description:
          carbon_source: "Carbon source in growth media"
        ```

    """

    factor_aliases: dict[str, dict[str, list[Any]]] = Field(
        default_factory=dict,
        description="Optional alias mappings for normalizing factor levels",
    )
    missing_value_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels for missing values by property name",
    )
    description: dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable descriptions for each property",
    )
    repositories: dict[str, RepositoryConfig] = Field(
        ..., description="Repository configurations keyed by repo ID"
    )

    @field_validator("missing_value_labels", mode="before")
    @classmethod
    def validate_missing_value_labels(cls, v: Any) -> dict[str, str]:
        """Validate missing value labels structure, filtering out None values."""
        if not v:
            return {}
        if not isinstance(v, dict):
            raise ValueError("missing_value_labels must be a dict")
        # Filter out None values that may come from empty YAML values
        return {k: val for k, val in v.items() if val is not None}

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, v: Any) -> dict[str, str]:
        """Validate description structure, filtering out None values."""
        if not v:
            return {}
        if not isinstance(v, dict):
            raise ValueError("description must be a dict")
        # Filter out None values that may come from empty YAML values
        return {k: val for k, val in v.items() if val is not None}

    @field_validator("factor_aliases")
    @classmethod
    def validate_factor_aliases(
        cls, v: dict[str, dict[str, list[Any]]]
    ) -> dict[str, dict[str, list[Any]]]:
        """Validate factor alias structure."""
        # Empty is OK - aliases are optional
        if not v:
            return v

        for prop_name, aliases in v.items():
            if not isinstance(aliases, dict):
                raise ValueError(
                    f"Property '{prop_name}' aliases must be a dict, "
                    f"got {type(aliases).__name__}"
                )

            # Validate each alias mapping
            for alias_name, actual_values in aliases.items():
                if not isinstance(actual_values, list):
                    raise ValueError(
                        f"Alias '{alias_name}' for '{prop_name}' must map "
                        f"to a list of values"
                    )
                if not actual_values:
                    raise ValueError(
                        f"Alias '{alias_name}' for '{prop_name}' cannot "
                        f"have empty value list"
                    )
                for val in actual_values:
                    if not isinstance(val, (str, int, float, bool)):
                        raise ValueError(
                            f"Alias '{alias_name}' for '{prop_name}' contains "
                            f"invalid value type: {type(val).__name__}"
                        )

        return v

    @model_validator(mode="before")
    @classmethod
    def parse_repositories(cls, data: Any) -> Any:
        """Parse repository configurations from 'repositories' key."""
        if not isinstance(data, dict):
            return data

        # Extract repositories from 'repositories' key
        repositories_data = data.get("repositories", {})

        if not repositories_data:
            raise ValueError(
                "Configuration must have a 'repositories' key with at least one repository"
            )

        if not isinstance(repositories_data, dict):
            raise ValueError("'repositories' key must contain a dict")

        repositories = {}
        for repo_id, repo_config in repositories_data.items():
            try:
                repositories[repo_id] = RepositoryConfig.model_validate(repo_config)
            except Exception as e:
                raise ValueError(
                    f"Invalid configuration for repository '{repo_id}': {e}"
                ) from e

        return {
            "factor_aliases": data.get("factor_aliases", {}),
            "missing_value_labels": data.get("missing_value_labels", {}),
            "description": data.get("description", {}),
            "repositories": repositories,
        }

    @classmethod
    def from_yaml(cls, path: Path | str) -> "MetadataConfig":
        """
        Load and validate configuration from YAML file.

        :param path: Path to YAML configuration file
        :return: Validated MetadataConfig instance
        :raises FileNotFoundError: If file doesn't exist
        :raises ValueError: If configuration is invalid

        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Configuration must be a YAML dict")

        return cls.model_validate(data)

    def get_repository_config(self, repo_id: str) -> RepositoryConfig | None:
        """
        Get configuration for a specific repository.

        :param repo_id: Repository ID (e.g., "BrentLab/harbison_2004")
        :return: RepositoryConfig instance or None if not found

        """
        return self.repositories.get(repo_id)

    def get_property_mappings(
        self, repo_id: str, config_name: str
    ) -> dict[str, PropertyMapping]:
        """
        Get merged property mappings for a repo/dataset combination.

        Merges repo-wide and dataset-specific mappings, with dataset-specific taking
        precedence.

        :param repo_id: Repository ID
        :param config_name: Dataset/config name
        :return: Dict mapping property names to PropertyMapping objects

        """
        repo_config = self.get_repository_config(repo_id)
        if not repo_config:
            return {}

        # Start with repo-wide properties
        mappings: dict[str, PropertyMapping] = dict(repo_config.properties)

        # Override with dataset-specific properties
        if repo_config.dataset and config_name in repo_config.dataset:
            mappings.update(repo_config.dataset[config_name])

        return mappings
