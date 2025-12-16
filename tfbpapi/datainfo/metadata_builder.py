"""
Metadata builder for creating standardized tables across heterogeneous datasets.

This module provides tools for building standardized metadata views by normalizing
factor levels across datasets with varying experimental condition structures. Users
specify optional alias mappings in external YAML files to standardize factor level
names (e.g., "D-glucose" -> "glucose").

Key Components:
- MetadataBuilder: Main class for building normalized metadata across datasets
- normalize_value(): Function for normalizing values using optional alias mappings
- Three output modes: conditions, samples, full_data
- External YAML configuration for aliases and property mappings

Example Configuration:
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
    ```

Example Usage:
    >>> builder = MetadataBuilder("metadata_config.yaml")
    >>> results = builder.build_metadata(
    ...     repos=[("BrentLab/harbison_2004", "harbison_2004")],
    ...     mode="conditions"
    ... )

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from tfbpapi.datainfo import DataCard
from tfbpapi.datainfo.metadata_config_models import MetadataConfig, PropertyMapping
from tfbpapi.errors import DataCardError
from tfbpapi.HfQueryAPI import HfQueryAPI


def get_nested_value(data: dict, path: str) -> Any:
    """
    Navigate nested dict using dot notation.

    Handles missing intermediate keys gracefully by returning None.

    :param data: Dictionary to navigate
    :param path: Dot-separated path (e.g., "media.carbon_source")
    :return: Value at path or None if not found

    """
    if not isinstance(data, dict):
        return None

    keys = path.split(".")
    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    return current


def extract_compound_names(value: Any) -> list[str]:
    """
    Extract compound names from various representations.

    Handles:
    - List of dicts: [{"compound": "D-glucose", ...}] -> ["D-glucose"]
    - String: "D-glucose" -> ["D-glucose"]
    - None or "unspecified" -> []

    :param value: Value to extract from
    :return: List of compound names

    """
    if value is None or value == "unspecified":
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        compounds = []
        for item in value:
            if isinstance(item, dict) and "compound" in item:
                compounds.append(item["compound"])
            elif isinstance(item, str):
                compounds.append(item)
        return compounds

    return []


def normalize_value(actual_value: Any, aliases: dict[str, list[Any]] | None) -> str:
    """
    Normalize a value using optional alias mappings (case-insensitive).

    Returns the alias name if a match is found, otherwise returns the
    original value as a string. This enables standardizing factor level
    names across heterogeneous datasets.

    :param actual_value: The value from the data to normalize
    :param aliases: Optional dict mapping alias names to lists of actual values.
                    Example: {"glucose": ["D-glucose", "dextrose"]}
    :return: Alias name if match found, otherwise str(actual_value)

    Examples:
        # With aliases - exact match
        normalize_value("D-glucose", {"glucose": ["D-glucose", "dextrose"]})
        -> "glucose"

        # With aliases - case-insensitive match
        normalize_value("DEXTROSE", {"glucose": ["D-glucose", "dextrose"]})
        -> "glucose"

        # No alias match - pass through
        normalize_value("maltose", {"glucose": ["D-glucose"]})
        -> "maltose"

        # No aliases provided - pass through
        normalize_value("D-glucose", None)
        -> "D-glucose"

        # Numeric value
        normalize_value(30, {"thirty": [30, "30"]})
        -> "thirty"

    """
    if aliases is None:
        return str(actual_value)

    # Convert to string for comparison (case-insensitive)
    actual_str = str(actual_value).lower()

    # Check each alias mapping
    for alias_name, actual_values in aliases.items():
        for val in actual_values:
            if str(val).lower() == actual_str:
                return alias_name

    # No match found - pass through original value
    return str(actual_value)


class MetadataBuilder:
    """
    Build standardized metadata tables across heterogeneous datasets.

    This class creates metadata views with normalized factor level names using
    optional alias mappings. Unlike a filtering system, this includes ALL data
    and simply normalizes the factor level names for standardization.

    Configuration specifies:
    1. Optional factor aliases (for normalizing factor level names)
    2. Repository and dataset-specific property paths (where to find each property)

    Configuration structure:
        factor_aliases:  # Optional
          property_name:
            alias_name: [actual_value1, actual_value2]

        BrentLab/repo_name:
          # Repo-wide properties (apply to all datasets in this repository)
          # Paths are relative to experimental_conditions at the repository level
          property1:
            path: media.name

          # Dataset-specific section
          dataset:
            dataset_name:
              # Dataset-specific properties (apply only to this dataset)
              # Paths are relative to experimental_conditions at the config level
              property2:
                path: temperature_celsius

              # Field-level properties (per-sample variation)
              # Paths are relative to field definitions (NOT experimental_conditions)
              property3:
                field: condition
                path: media.carbon_source

    Path Resolution:
      - Repo-wide & dataset-specific: Paths automatically
        prepended with "experimental_conditions."
        Example: path "media.name" resolves to experimental_conditions.media.name

      - Field-level: Paths used directly on field definitions (no prepending)
        Example: field "condition", path "media.carbon_source"
                 looks in condition field's definitions for media.carbon_source

    Three output modes:
    - conditions: Extract normalized metadata (no data retrieval)
    - samples: Sample-level metadata (one row per sample_id)
    - full_data: All measurements with metadata

    Attributes:
        config: Validated MetadataConfig instance
        factor_aliases: Optional alias mappings for normalization

    """

    def __init__(self, config_path: Path | str):
        """
        Initialize metadata builder with external configuration.

        :param config_path: Path to YAML configuration file
        :raises FileNotFoundError: If config file doesn't exist
        :raises ValueError: If configuration is invalid

        """
        self.config = MetadataConfig.from_yaml(config_path)
        self.factor_aliases = self.config.factor_aliases

    def _get_property_mappings(
        self, repo_id: str, config_name: str
    ) -> dict[str, PropertyMapping]:
        """
        Get property mappings for a specific repo/dataset combination.

        Merges repo-wide and dataset-specific mappings, with dataset-specific taking
        precedence.

        :param repo_id: Repository ID
        :param config_name: Dataset/config name
        :return: Dict mapping property names to PropertyMapping objects

        """
        return self.config.get_property_mappings(repo_id, config_name)

    def build_metadata(
        self,
        repos: list[tuple[str, str]],
        mode: str = "conditions",
        token: str | None = None,
    ) -> dict[str, Any]:
        """
        Build metadata tables across datasets with normalized factor levels.

        Note: ALL repositories are processed - no filtering/exclusion occurs.
        Factor aliases are used to normalize factor level names, not to filter.

        :param repos: List of (repo_id, config_name) tuples to process
        :param mode: Output mode - "conditions", "samples", or "full_data"
        :param token: Optional HuggingFace token for private repos
        :return: Dict mapping repo_id to metadata results
        :raises ValueError: If mode is invalid

        """
        if mode not in ["conditions", "samples", "full_data"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'conditions', "
                "'samples', or 'full_data'"
            )

        results = {}

        for repo_id, config_name in repos:
            try:
                # Load DataCard
                card = DataCard(repo_id, token=token)

                # Check if this repository has configuration
                if repo_id not in self.config.repositories:
                    results[repo_id] = {
                        "error": f"No property mappings defined for {repo_id}"
                    }
                    continue

                # Extract and normalize metadata for ALL data
                metadata = self._extract_metadata(card, repo_id, config_name)

                # Build result based on mode
                result = {"metadata": metadata}

                if mode == "conditions":
                    # Mode 0: Just metadata, no data
                    pass
                elif mode == "samples":
                    # Mode 1: Sample-level metadata
                    result["data"] = self._get_sample_metadata(
                        repo_id, config_name, metadata, token
                    )
                elif mode == "full_data":
                    # Mode 2: Full data with measurements
                    result["data"] = self._get_full_data(
                        repo_id, config_name, metadata, token
                    )

                results[repo_id] = result

            except Exception as e:
                results[repo_id] = {"error": f"Error processing dataset: {str(e)}"}

        return results

    def _extract_metadata(
        self,
        card: DataCard,
        repo_id: str,
        config_name: str,
    ) -> dict[str, Any]:
        """
        Extract and normalize metadata from datacard.

        Extracts ALL metadata with normalized factor level names.
        No filtering occurs - all data is included.

        :param card: DataCard instance
        :param repo_id: Repository ID
        :param config_name: Configuration name
        :return: Dict with normalized metadata

        """
        metadata = {}
        property_mappings = self._get_property_mappings(repo_id, config_name)

        # Extract repo/config level metadata
        repo_metadata = self._extract_repo_level(card, config_name, property_mappings)
        metadata.update(repo_metadata)

        # Extract field-level metadata
        field_metadata = self._extract_field_level(card, config_name, property_mappings)
        if field_metadata:
            metadata["field_values"] = field_metadata

        return metadata

    def _extract_repo_level(
        self,
        card: DataCard,
        config_name: str,
        property_mappings: dict[str, PropertyMapping],
    ) -> dict[str, list[str]]:
        """
        Extract and normalize repo/config-level metadata.

        :param card: DataCard instance
        :param config_name: Configuration name
        :param property_mappings: Property mappings for this repo/dataset
        :return: Dict mapping property names to normalized values

        """
        metadata = {}

        # Get repo and config level conditions
        try:
            conditions = card.get_experimental_conditions(config_name)
        except DataCardError:
            conditions = {}

        if not conditions:
            return metadata

        # Extract each mapped property
        for prop_name, mapping in property_mappings.items():
            # Skip field-level mappings (handled separately)
            if mapping.field is not None:
                continue

            # Get path for this property
            path = mapping.path
            full_path = f"experimental_conditions.{path}"

            # Get value at this path
            value = get_nested_value(conditions, full_path)

            if value is None:
                continue

            # Extract compound names if needed
            if isinstance(value, list) and value and isinstance(value[0], dict):
                actual_values = extract_compound_names(value)
            else:
                actual_values = [value] if not isinstance(value, list) else value

            # Normalize using aliases (if configured)
            aliases = self.factor_aliases.get(prop_name)
            normalized_values = [normalize_value(v, aliases) for v in actual_values]

            metadata[prop_name] = normalized_values

        return metadata

    def _extract_field_level(
        self,
        card: DataCard,
        config_name: str,
        property_mappings: dict[str, PropertyMapping],
    ) -> dict[str, dict[str, Any]]:
        """
        Extract and normalize field-level metadata.

        Returns metadata for ALL field values (no filtering).

        :param card: DataCard instance
        :param config_name: Configuration name
        :param property_mappings: Property mappings for this repo/dataset
        :return: Dict mapping field values to their normalized metadata

        """
        field_metadata: dict[str, dict[str, Any]] = {}

        # Group property mappings by field
        field_mappings: dict[str, dict[str, str]] = {}
        for prop_name, mapping in property_mappings.items():
            if mapping.field is not None:
                field_name = mapping.field
                if field_name not in field_mappings:
                    field_mappings[field_name] = {}
                field_mappings[field_name][prop_name] = mapping.path

        # Process each field that has mappings
        for field_name, prop_paths in field_mappings.items():
            # Get field definitions
            definitions = card.get_field_definitions(config_name, field_name)
            if not definitions:
                continue

            # Extract metadata for ALL field values (no filtering!)
            for field_value, definition in definitions.items():
                if field_value not in field_metadata:
                    field_metadata[field_value] = {}

                for prop_name, path in prop_paths.items():
                    # Get value at path
                    value = get_nested_value(definition, path)

                    if value is None:
                        continue

                    # Extract compound names if needed
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        actual_values = extract_compound_names(value)
                    else:
                        actual_values = (
                            [value] if not isinstance(value, list) else value
                        )

                    # Normalize using aliases (if configured)
                    aliases = self.factor_aliases.get(prop_name)
                    normalized_values = [
                        normalize_value(v, aliases) for v in actual_values
                    ]

                    field_metadata[field_value][prop_name] = normalized_values

        return field_metadata

    def _get_sample_metadata(
        self,
        repo_id: str,
        config_name: str,
        metadata: dict[str, Any],
        token: str | None = None,
    ) -> pd.DataFrame:
        """
        Get sample-level metadata (Mode 1).

        Returns one row per sample_id with metadata columns. Includes ALL samples (no
        filtering).

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :param metadata: Extracted metadata dict
        :param token: Optional HuggingFace token
        :return: DataFrame with sample metadata

        """
        # Initialize query API
        api = HfQueryAPI(repo_id, token=token)

        # Query for all samples (no WHERE clause filtering)
        sql = f"""
            SELECT DISTINCT *
            FROM {config_name}
        """

        try:
            df = api.query(sql, config_name)

            # For sample-level, we want one row per sample_id
            if "sample_id" in df.columns:
                df_samples = df.groupby("sample_id").first().reset_index()
                return df_samples
            else:
                # No sample_id column, return distinct rows
                return df

        except Exception as e:
            return pd.DataFrame(
                {"error": [str(e)], "note": ["Failed to retrieve sample metadata"]}
            )

    def _get_full_data(
        self,
        repo_id: str,
        config_name: str,
        metadata: dict[str, Any],
        token: str | None = None,
    ) -> pd.DataFrame:
        """
        Get full data with all measurements (Mode 2).

        Returns many rows per sample_id (one per measured feature/target). Includes ALL
        data (no filtering).

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :param metadata: Extracted metadata dict
        :param token: Optional HuggingFace token
        :return: DataFrame with full data

        """
        # Initialize query API
        api = HfQueryAPI(repo_id, token=token)

        # Query for all data (no WHERE clause filtering)
        sql = f"""
            SELECT *
            FROM {config_name}
        """

        try:
            return api.query(sql, config_name)
        except Exception as e:
            return pd.DataFrame(
                {"error": [str(e)], "note": ["Failed to retrieve full data"]}
            )

    def __repr__(self) -> str:
        """String representation."""
        n_props = len(self.factor_aliases)
        n_repos = len(self.config.repositories)
        return (
            f"MetadataBuilder({n_props} properties with aliases, "
            f"{n_repos} repositories configured)"
        )
