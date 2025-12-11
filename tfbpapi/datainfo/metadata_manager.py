"""
MetadataManager for creating metadata table views from DataCard information.

This module provides the MetadataManager class for generating SQL commands that create
metadata views from HuggingFace dataset information extracted via DataCard. The manager
enables users to:

- Specify which columns to include in metadata views
- Add constant columns from top-level or config-level experimental conditions
- Flatten nested condition definitions into queryable columns
- Generate SQL CREATE VIEW statements

The design follows a user-driven workflow:
1. User explores DataCard to understand available fields and conditions
2. User downloads data via HfCacheManager/HfQueryAPI (creates base views)
3. User specifies column selection for metadata view
4. MetadataManager generates and executes SQL to create the view

"""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# Separator conventions for formatting compound metadata values
COMPONENT_SEPARATORS = {
    "type_value": ":",  # Separates type from value (e.g., "type:value")
    "value_conc": "@",  # Separates value from concentration (e.g., "compound@0.5%")
    "components": ";",  # Separates multiple components (e.g., "comp1;comp2;comp3")
    "types": "|",  # Separates component types (e.g., "type1|type2|type3")
}


class MetadataManager:
    """
    Manager for creating metadata table views from DataCard information.

    MetadataManager provides a flexible interface for generating SQL views that combine
    data fields with experimental condition metadata. Users specify which columns to
    include and how to handle conditions at different hierarchy levels.

    Example:
        >>> # Step 1: Explore schema with DataCard
        >>> card = DataCard("BrentLab/harbison_2004")
        >>> schema = card.extract_metadata_schema("harbison_2004")
        >>>
        >>> # Step 2: Create metadata view
        >>> mgr = MetadataManager()
        >>> view = mgr.create_metadata_view(
        ...     base_view="harbison_2004_train",
        ...     view_name="harbison_2004_metadata",
        ...     include_fields=["regulator_locus_tag", "target_locus_tag", "condition"],
        ...     constant_columns={
        ...         "temperature_celsius": 30,
        ...         "strain_background": "BY4741"
        ...     }
        ... )

    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache: bool = False,
        duckdb_conn: duckdb.DuckDBPyConnection | None = None,
    ):
        """
        Initialize MetadataManager.

        :param cache_dir: Optional directory for caching metadata views
        :param cache: Whether to enable persistent caching (not yet implemented)
        :param duckdb_conn: Optional DuckDB connection to use. If None, creates new
            in-memory connection. Pass a shared connection to work with views created
            by HfQueryAPI or other tools.

        """
        self._conn = duckdb_conn if duckdb_conn is not None else duckdb.connect(":memory:")
        self._cache_dir = cache_dir
        self._cache_enabled = cache
        self._registered_datasets: dict[str, Any] = {}
        self._view_names: dict[tuple[str, str], str] = {}

    def _sanitize_view_name(self, repo_id: str, config_name: str) -> str:
        """
        Convert repo_id and config_name to valid SQL identifier.

        Replaces special characters (/, -, spaces) with underscores and appends
        '_metadata' suffix.

        :param repo_id: Repository ID (e.g., "BrentLab/harbison_2004")
        :param config_name: Configuration name (e.g., "harbison_2004")
        :return: Sanitized view name (e.g., "BrentLab_harbison_2004_harbison_2004_metadata")

        Example:
            >>> mgr = MetadataManager()
            >>> mgr._sanitize_view_name("BrentLab/dataset-name", "config_name")
            'BrentLab_dataset_name_config_name_metadata'

        """
        sanitized = f"{repo_id}_{config_name}"
        sanitized = sanitized.replace("/", "_").replace("-", "_").replace(" ", "_")
        return f"{sanitized}_metadata"

    def _flatten_condition_definition(self, definition: dict) -> dict:
        """
        Flatten nested condition definition dict into flat key-value pairs.

        Extracts common experimental condition fields:
        - media.name -> growth_media
        - temperature_celsius -> temperature_celsius
        - cultivation_method -> cultivation_method
        - strain_background -> strain_background

        :param definition: Nested definition dict from field.definitions[value]
        :return: Flat dict with standardized keys

        Example:
            >>> mgr = MetadataManager()
            >>> definition = {
            ...     "media": {"name": "YPD"},
            ...     "temperature_celsius": 30
            ... }
            >>> mgr._flatten_condition_definition(definition)
            {'growth_media': 'YPD', 'temperature_celsius': 30}

        """
        result = {}
        if not definition:
            return result

        # Extract media name as growth_media
        if "media" in definition and isinstance(definition["media"], dict):
            if "name" in definition["media"]:
                result["growth_media"] = definition["media"]["name"]

        # Extract other top-level condition fields
        for key in ["temperature_celsius", "cultivation_method", "strain_background"]:
            if key in definition:
                result[key] = definition[key]

        return result

    def _flatten_experimental_conditions(self, exp_conds: Any) -> dict:
        """
        Extract attributes from ExperimentalConditions object or dict.

        Handles both Pydantic model objects and plain dicts. Extracts:
        - temperature_celsius
        - cultivation_method
        - strain_background
        - media.name (as growth_media)

        :param exp_conds: ExperimentalConditions object or dict
        :return: Flat dict with condition values

        Example:
            >>> mgr = MetadataManager()
            >>> class MockConditions:
            ...     temperature_celsius = 30
            ...     strain_background = "BY4741"
            >>> mgr._flatten_experimental_conditions(MockConditions())
            {'temperature_celsius': 30, 'strain_background': 'BY4741'}

        """
        result = {}
        if exp_conds is None:
            return result

        # Handle both objects (with attributes) and dicts (with keys)
        for attr in ["temperature_celsius", "cultivation_method", "strain_background"]:
            if hasattr(exp_conds, attr):
                val = getattr(exp_conds, attr, None)
            else:
                val = exp_conds.get(attr) if isinstance(exp_conds, dict) else None

            if val is not None:
                result[attr] = val

        # Extract media.name if present
        if hasattr(exp_conds, "media"):
            media = getattr(exp_conds, "media", None)
        else:
            media = exp_conds.get("media") if isinstance(exp_conds, dict) else None

        if media:
            if hasattr(media, "name"):
                name = getattr(media, "name", None)
            else:
                name = media.get("name") if isinstance(media, dict) else None

            if name:
                result["growth_media"] = name

        return result

    def create_metadata_view(
        self,
        base_view: str,
        view_name: str,
        include_fields: list[str],
        constant_columns: dict[str, Any] | None = None,
    ) -> str:
        """
        Create metadata view from base view with user-specified columns.

        Generates SQL CREATE OR REPLACE VIEW statement that selects specified fields
        from an existing base view and adds constant columns as literals.

        :param base_view: Name of existing DuckDB view (created by HfCacheManager)
        :param view_name: Name for the new metadata view
        :param include_fields: List of field names to select from base view
        :param constant_columns: Optional dict of {column_name: value} to add as constants
        :return: Name of created view

        Example:
            >>> mgr = MetadataManager()
            >>> view = mgr.create_metadata_view(
            ...     base_view="harbison_2004_train",
            ...     view_name="harbison_2004_metadata",
            ...     include_fields=["regulator_locus_tag", "target_locus_tag"],
            ...     constant_columns={"temperature_celsius": 30}
            ... )
            >>> view
            'harbison_2004_metadata'

        """
        # Build SELECT clause with included fields
        select_parts = include_fields.copy()

        # Add constant columns as literals
        if constant_columns:
            for col_name, col_value in constant_columns.items():
                if isinstance(col_value, str):
                    select_parts.append(f"'{col_value}' AS {col_name}")
                else:
                    select_parts.append(f"{col_value} AS {col_name}")

        select_clause = ",\n    ".join(select_parts)

        # Generate SQL
        sql = f"""\
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT {select_clause}
            FROM {base_view}
"""

        # Execute
        self._conn.execute(sql)

        return view_name

    def get_active_configs(self) -> list[tuple[str, str]]:
        """
        Get list of registered (repo_id, config_name) tuples.

        :return: List of active config tuples (empty for minimal implementation)

        Note:
            This is a placeholder method for future multi-dataset support.
            Currently returns empty list.

        """
        return []

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of registered datasets.

        :return: Empty DataFrame for minimal implementation

        Note:
            This is a placeholder method for future multi-dataset support.
            Currently returns empty DataFrame.

        """
        return pd.DataFrame()
