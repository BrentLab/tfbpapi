"""
Dataset filter resolver with external configuration for heterogeneous datasets.

This module provides a simple, configuration-driven approach to filtering samples
across datasets with varying experimental condition structures. Users specify
filters and dataset-specific property paths in external YAML files.

Key Components:
- DatasetFilterResolver: Main class for applying filters across datasets
- Three output modes: conditions, samples, full_data
- External YAML configuration for filters and property mappings
- Automatic detection of property location (repo/config/field level)

Example Configuration:
    ```yaml
    filters:
      carbon_source: ["D-glucose", "D-galactose"]
      temperature_celsius: [30, 37]

    dataset_mappings:
      "BrentLab/harbison_2004":
        carbon_source:
          path: "environmental_conditions.media.carbon_source"
        temperature_celsius:
          path: "environmental_conditions.temperature_celsius"
    ```

Example Usage:
    >>> resolver = DatasetFilterResolver("filters.yaml")
    >>> results = resolver.resolve_filters(
    ...     repos=[("BrentLab/harbison_2004", "harbison_2004")],
    ...     mode="conditions"
    ... )

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

from tfbpapi.datainfo import DataCard
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


class DatasetFilterResolver:
    """
    Resolve filters across heterogeneous datasets using external configuration.

    This class takes an external YAML configuration specifying:
    1. Filter criteria (which values are acceptable for each property)
    2. Repository and dataset-specific property paths (where to find each property)

    Configuration structure:
        filters:
          property_name: [value1, value2]

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

    Examples:
        # Repo-wide property (applies to all datasets in BrentLab/kemmeren_2014)
        filters:
          temperature_celsius: [30]

        BrentLab/kemmeren_2014:
          temperature_celsius:
            path: temperature_celsius

        # Dataset-specific property (applies only to kemmeren_2014 dataset)
        BrentLab/kemmeren_2014:
          dataset:
            kemmeren_2014:
              carbon_source:
                path: media.carbon_source

        # Field-level property (per-sample variation via condition field)
        BrentLab/harbison_2004:
          dataset:
            harbison_2004:
              carbon_source:
                field: condition
                path: media.carbon_source

    It then resolves filters across datasets with three output modes:
    - conditions: Just which field values match (no data retrieval)
    - samples: Sample-level metadata (one row per sample_id)
    - full_data: All measurements for matching samples

    Attributes:
        config: Loaded configuration dict
        filters: Filter criteria from config
        mappings: Repository/dataset-specific property paths

    """

    def __init__(self, config_path: Path | str):
        """
        Initialize resolver with external configuration.

        :param config_path: Path to YAML configuration file

        """
        self.config = self._load_config(Path(config_path))
        self.filters = self.config.get("filters", {})
        # Extract mappings: all keys except 'filters' are repository IDs
        self.mappings = {k: v for k, v in self.config.items() if k != "filters"}

    def _load_config(self, config_path: Path) -> dict:
        """
        Load YAML configuration file.

        :param config_path: Path to YAML file
        :return: Configuration dict
        :raises FileNotFoundError: If config file doesn't exist
        :raises yaml.YAMLError: If config is invalid YAML

        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dict")

        return config

    def _get_property_mappings(self, repo_id: str, config_name: str) -> dict:
        """
        Get property mappings for a specific repo/dataset combination.

        Merges repo-wide and dataset-specific mappings,
        with dataset-specific taking precedence.

        Configuration format:
            BrentLab/repo_name:
              # Repo-wide properties (apply to all datasets)
              property1:
                path: path.in.experimental_conditions

              # Dataset-specific section
              dataset:
                dataset_name:
                  property2:
                    field: field_name  # For field-level definitions
                    path: path.within.field

        Examples:
          1. Field-specific path (for field-level definitions):
             carbon_source:
               field: condition
               path: media.carbon_source

          2. Repo-level path (for experimental_conditions):
             temperature:
               path: temperature_celsius

        :param repo_id: Repository ID
        :param config_name: Dataset/config name
        :return: Merged property mappings dict

        """
        if repo_id not in self.mappings:
            return {}

        repo_config = self.mappings[repo_id]
        mappings = {}

        # Add repo-wide properties (all keys except 'dataset')
        for key, value in repo_config.items():
            if key != "dataset":
                mappings[key] = value

        # Override with dataset-specific properties
        if "dataset" in repo_config:
            if config_name in repo_config["dataset"]:
                mappings.update(repo_config["dataset"][config_name])

        return mappings

    def resolve_filters(
        self,
        repos: list[tuple[str, str]],
        mode: str = "conditions",
        token: str | None = None,
    ) -> dict[str, Any]:
        """
        Resolve filters across datasets.

        :param repos: List of (repo_id, config_name) tuples to check
        :param mode: Output mode - "conditions", "samples", or "full_data"
        :param token: Optional HuggingFace token for private repos
        :return: Dict mapping repo_id to results
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

                # Check if this dataset has mappings
                if repo_id not in self.mappings:
                    results[repo_id] = {
                        "included": False,
                        "reason": f"No property mappings defined for {repo_id}",
                    }
                    continue

                # Check repo/config level filters (Level 1)
                included, reason = self._check_repo_config_level(
                    card, config_name, repo_id
                )

                if not included:
                    results[repo_id] = {"included": False, "reason": reason}
                    continue

                # Dataset passes Level 1, check field-level filters (Level 2)
                matching_field_values = self._check_field_level(
                    card, config_name, repo_id
                )

                # Build result based on mode
                result = {
                    "included": True,
                    "matching_field_values": matching_field_values,
                }

                if mode == "conditions":
                    # Mode 0: Just conditions, no data
                    pass
                elif mode == "samples":
                    # Mode 1: Sample-level metadata
                    result["data"] = self._get_sample_metadata(
                        repo_id, config_name, matching_field_values, token
                    )
                elif mode == "full_data":
                    # Mode 2: Full data with measurements
                    result["data"] = self._get_full_data(
                        repo_id, config_name, matching_field_values, token
                    )

                results[repo_id] = result

            except Exception as e:
                results[repo_id] = {
                    "included": False,
                    "reason": f"Error processing dataset: {str(e)}",
                }

        return results

    def _check_repo_config_level(
        self,
        card: DataCard,
        config_name: str,
        repo_id: str,
    ) -> tuple[bool, str]:
        """
        Check if repo/config level conditions match filters (Level 1).

        :param card: DataCard instance
        :param config_name: Configuration name
        :param repo_id: Repository ID
        :return: (included, reason) tuple

        """
        # Get repo and config level conditions
        try:
            conditions = card.get_experimental_conditions(config_name)
        except DataCardError:
            conditions = {}

        if not conditions:
            # No repo/config conditions to check, include by default
            return True, ""

        # Check each filter property at repo/config level
        property_mappings = self._get_property_mappings(repo_id, config_name)

        for filter_prop, acceptable_values in self.filters.items():
            if filter_prop not in property_mappings:
                continue

            mapping = property_mappings[filter_prop]

            # Check if this is a field-level mapping
            # (should not be checked at repo level)
            if "field" in mapping:
                # Skip field-level properties at repo/config level
                continue

            # Get path for this property
            path = mapping["path"]

            # Prepend experimental_conditions for repo-level paths
            full_path = f"experimental_conditions.{path}"

            # Try to get value at this path
            value = get_nested_value(conditions, full_path)

            if value is None:
                # Property not specified at repo/config level, skip
                continue

            # Extract compound names if this is a compound list
            if isinstance(value, list) and value and isinstance(value[0], dict):
                actual_values = extract_compound_names(value)
            else:
                actual_values = [value] if not isinstance(value, list) else value

            # Check if any actual value matches acceptable values
            matches = False
            for actual in actual_values:
                if actual in acceptable_values:
                    matches = True
                    break

            if not matches:
                return (
                    False,
                    f"{filter_prop}: found {actual_values}, "
                    f"wanted {acceptable_values}",
                )

        return True, ""

    def _check_field_level(
        self,
        card: DataCard,
        config_name: str,
        repo_id: str,
    ) -> dict[str, list[str]]:
        """
        Check field-level conditions and return matching field values (Level 2).

        :param card: DataCard instance
        :param config_name: Configuration name
        :param repo_id: Repository ID
        :return: Dict mapping field names to lists of matching values

        """
        matching: dict[str, list[str]] = {}
        property_mappings = self._get_property_mappings(repo_id, config_name)

        # Get config to find fields with role=experimental_condition
        config = card.get_config(config_name)
        if not config:
            return matching

        # Group property mappings by field (if field is specified)
        # field_mappings: {field_name: {prop: path, ...}}
        field_mappings: dict[str, dict[str, str]] = {}
        general_mappings: dict[str, str] = {}  # Properties without field specification

        for prop, mapping in property_mappings.items():
            if "field" in mapping:
                # Field-specific mapping
                field_name = mapping["field"]
                if field_name not in field_mappings:
                    field_mappings[field_name] = {}
                field_mappings[field_name][prop] = mapping["path"]
            else:
                # General mapping (repo/config level)
                general_mappings[prop] = mapping["path"]

        # Process each field that has mappings
        for field_name, prop_paths in field_mappings.items():
            # Check if this field has definitions
            definitions = card.get_field_definitions(config_name, field_name)
            if not definitions:
                continue

            matching_values = []

            for field_value, definition in definitions.items():
                # Check if this field value matches all filter criteria
                matches_all = True

                for filter_prop, acceptable_values in self.filters.items():
                    if filter_prop not in prop_paths:
                        continue

                    # Get path for this property
                    path = prop_paths[filter_prop]

                    # Get value from this field value's definition
                    value = get_nested_value(definition, path)

                    if value is None:
                        # Property not in this definition, doesn't match
                        matches_all = False
                        break

                    # Extract compound names if needed
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        actual_values = extract_compound_names(value)
                    else:
                        actual_values = (
                            [value] if not isinstance(value, list) else value
                        )

                    # Check if any actual value matches acceptable values
                    matches = False
                    for actual in actual_values:
                        if actual in acceptable_values:
                            matches = True
                            break

                    if not matches:
                        matches_all = False
                        break

                if matches_all:
                    matching_values.append(field_value)

            if matching_values:
                matching[field_name] = matching_values

        return matching

    def _get_sample_metadata(
        self,
        repo_id: str,
        config_name: str,
        matching_field_values: dict[str, list[str]],
        token: str | None = None,
    ) -> pd.DataFrame:
        """
        Get sample-level metadata (Mode 1).

        Returns one row per sample_id with metadata columns.

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :param matching_field_values: Dict of field names to matching values
        :param token: Optional HuggingFace token
        :return: DataFrame with sample metadata

        """
        # Initialize query API
        api = HfQueryAPI(repo_id, token=token)

        # Build WHERE clause from matching field values
        where_conditions = []
        for field_name, values in matching_field_values.items():
            if len(values) == 1:
                # Single value: exact match
                where_conditions.append(f"{field_name} = '{values[0]}'")
            else:
                # Multiple values: IN clause
                values_str = "', '".join(values)
                where_conditions.append(f"{field_name} IN ('{values_str}')")

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Query for distinct sample_id with metadata fields
        # Get all columns to understand structure
        sql = f"""
            SELECT DISTINCT *
            FROM {config_name}
            WHERE {where_clause}
        """

        try:
            df = api.query(sql, config_name)

            # For sample-level, we want one row per sample_id
            # Group by sample_id and take first value for other columns
            if "sample_id" in df.columns:
                # Get metadata columns (exclude measurement columns if possible)
                # This is a heuristic - may need refinement
                df_samples = df.groupby("sample_id").first().reset_index()
                return df_samples
            else:
                # No sample_id column, return distinct rows
                return df

        except Exception as e:
            # Return error info
            return pd.DataFrame(
                {"error": [str(e)], "note": ["Failed to retrieve sample metadata"]}
            )

    def _get_full_data(
        self,
        repo_id: str,
        config_name: str,
        matching_field_values: dict[str, list[str]],
        token: str | None = None,
    ) -> pd.DataFrame:
        """
        Get full data with all measurements (Mode 2).

        Returns many rows per sample_id (one per measured feature/target).

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :param matching_field_values: Dict of field names to matching values
        :param token: Optional HuggingFace token
        :return: DataFrame with full data

        """
        # Initialize query API
        api = HfQueryAPI(repo_id, token=token)

        # Build WHERE clause from matching field values
        where_conditions = []
        for field_name, values in matching_field_values.items():
            if len(values) == 1:
                # Single value: exact match
                where_conditions.append(f"{field_name} = '{values[0]}'")
            else:
                # Multiple values: IN clause
                values_str = "', '".join(values)
                where_conditions.append(f"{field_name} IN ('{values_str}')")

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Query for all data matching the conditions
        sql = f"""
            SELECT *
            FROM {config_name}
            WHERE {where_clause}
        """

        try:
            return api.query(sql, config_name)
        except Exception as e:
            # Return error info
            return pd.DataFrame(
                {"error": [str(e)], "note": ["Failed to retrieve full data"]}
            )

    def __repr__(self) -> str:
        """String representation."""
        n_filters = len(self.filters)
        n_datasets = len(self.mappings)
        return (
            f"DatasetFilterResolver({n_filters} filters, "
            f"{n_datasets} datasets configured)"
        )
