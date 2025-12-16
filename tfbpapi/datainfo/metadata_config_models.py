"""
Pydantic models for metadata normalization configuration.

This module defines the schema for MetadataBuilder configuration files, providing
validation for factor alias mappings and repository configurations.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator, model_validator


class PropertyMapping(BaseModel):
    """
    Mapping specification for a single property.

    Attributes:
        path: Dot-notation path to the property value.
              For repo/config-level: relative to experimental_conditions
              For field-level: relative to field definitions
        field: Optional field name for field-level properties.
               When specified, looks in this field's definitions.
               When omitted, looks in repo/config-level experimental_conditions.

    Examples:
        Field-level property:
            PropertyMapping(field="condition", path="media.carbon_source")

        Repo/config-level property:
            PropertyMapping(path="temperature_celsius")

    """

    field: str | None = Field(None, description="Field name for field-level properties")
    path: str = Field(..., min_length=1, description="Dot-notation path to property")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path is not just whitespace."""
        if not v.strip():
            raise ValueError("path cannot be empty or whitespace")
        return v.strip()

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str | None) -> str | None:
        """Ensure field is not empty string if provided."""
        if v is not None and not v.strip():
            raise ValueError("field cannot be empty or whitespace")
        return v.strip() if v else None


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
        repositories: Dict mapping repository IDs to their configurations

    Example:
        ```yaml
        factor_aliases:
          carbon_source:
            glucose: ["D-glucose", "dextrose"]
            galactose: ["D-galactose", "Galactose"]

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
        ```

    """

    factor_aliases: dict[str, dict[str, list[Any]]] = Field(
        default_factory=dict,
        description="Optional alias mappings for normalizing factor levels",
    )
    repositories: dict[str, RepositoryConfig] = Field(
        ..., description="Repository configurations keyed by repo ID"
    )

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
        """Parse repository configurations from top-level keys."""
        if not isinstance(data, dict):
            return data

        # Extract repositories (all keys except 'factor_aliases')
        repositories = {}
        for key, value in data.items():
            if key != "factor_aliases":
                try:
                    repositories[key] = RepositoryConfig.model_validate(value)
                except Exception as e:
                    raise ValueError(
                        f"Invalid configuration for repository '{key}': {e}"
                    ) from e

        if not repositories:
            raise ValueError(
                "Configuration must have at least one repository configuration"
            )

        return {
            "factor_aliases": data.get("factor_aliases", {}),
            "repositories": repositories,
        }

    @classmethod
    def from_yaml(cls, path: Path | str) -> MetadataConfig:
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
