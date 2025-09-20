import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from .constants import CACHE_DIR
from .HfCacheManager import HfCacheManager


class HfQueryAPI(HfCacheManager):
    """Minimal Hugging Face API client focused on metadata retrieval."""

    def __init__(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
        token: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        """
        Initialize the minimal HF Query API client.

        :param repo_id: Repository identifier (e.g., "user/dataset")
        :param repo_type: Type of repository ("dataset", "model", "space")
        :param token: HuggingFace token for authentication
        :param cache_dir: HF cache_dir for downloads

        """
        # No DuckDB connection needed for metadata-only functionality
        import duckdb

        self._duckdb_conn = duckdb.connect(":memory:")

        # Initialize parent with minimal setup
        super().__init__(
            repo_id=repo_id,
            duckdb_conn=self._duckdb_conn,
            token=token,
            logger=logging.getLogger(self.__class__.__name__),
        )

        # Store basic configuration
        self.repo_type = repo_type
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR

        # Filter storage system
        self._table_filters: dict[str, str] = {}  # config_name -> SQL WHERE clause

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: str | Path) -> None:
        """Set the cache directory for huggingface_hub downloads."""
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Cache directory {path} does not exist")
        self._cache_dir = path

    def get_metadata(self, config_name: str | None = None) -> pd.DataFrame:
        """
        Retrieve metadata as a DataFrame with actual metadata values.

        For explicit metadata (dataset_type == METADATA): Returns all rows from metadata
        table. For embedded metadata (has metadata_fields): Returns distinct
        combinations of metadata fields.

        :param config_name: Optional specific config name, otherwise returns all
            metadata
        :return: DataFrame with metadata values
        :raises ValueError: If config_name is specified but not found

        """
        # Get explicit metadata configurations
        explicit_metadata_configs = self.dataset_card.get_metadata_configs()

        # Get data configurations that have embedded metadata
        # (metadata_fields specified)
        embedded_metadata_configs = [
            config
            for config in self.dataset_card.get_data_configs()
            if config.metadata_fields
        ]

        # Combine both types
        all_metadata_sources = explicit_metadata_configs + embedded_metadata_configs

        if not all_metadata_sources:
            # Return empty DataFrame
            return pd.DataFrame()

        # Filter by config_name if specified
        if config_name:
            matching_configs = [
                c for c in all_metadata_sources if c.config_name == config_name
            ]
            if not matching_configs:
                available_names = [c.config_name for c in all_metadata_sources]
                raise ValueError(
                    f"Config '{config_name}' not found. "
                    f"Available metadata configs: {available_names}"
                )
            configs_to_process = matching_configs
        else:
            configs_to_process = all_metadata_sources

        # Process each configuration and collect DataFrames
        dataframes = []
        for config in configs_to_process:
            # Ensure the data/metadata is loaded
            config_result = self._get_metadata_for_config(config)

            if not config_result.get("success", False):
                self.logger.warning(
                    f"Failed to load data for config {config.config_name}"
                )
                continue

            table_name = config_result.get("table_name")
            if not table_name:
                self.logger.warning(f"No table name for config {config.config_name}")
                continue

            try:
                if config in explicit_metadata_configs:
                    # Explicit metadata: return all rows from metadata table
                    sql = f"SELECT * FROM {table_name}"
                else:
                    # Embedded metadata: return distinct combinations of metadata fields
                    if config.metadata_fields is None:
                        raise ValueError(
                            f"Config {config.config_name} has no metadata fields"
                        )
                    fields = ", ".join(config.metadata_fields)
                    where_clauses = " AND ".join(
                        [f"{field} IS NOT NULL" for field in config.metadata_fields]
                    )
                    sql = f"""
                        SELECT DISTINCT {fields}, COUNT(*) as count
                        FROM {table_name}
                        WHERE {where_clauses}
                        GROUP BY {fields}
                        ORDER BY count DESC
                    """

                df = self.duckdb_conn.execute(sql).fetchdf()

                # Add config source column if multiple configs
                if len(configs_to_process) > 1:
                    df["config_name"] = config.config_name

                dataframes.append(df)

            except Exception as e:
                self.logger.error(
                    f"Error querying metadata for {config.config_name}: {e}"
                )
                continue

        # Combine all DataFrames
        if not dataframes:
            return pd.DataFrame()
        elif len(dataframes) == 1:
            return dataframes[0]
        else:
            return pd.concat(dataframes, ignore_index=True)

    def set_filter(self, config_name: str, **kwargs) -> None:
        """
        Set simple filters using keyword arguments.

        Converts keyword arguments to SQL WHERE clause and stores
        for automatic application.

        :param config_name: Configuration name to apply filters to
        :param kwargs: Filter conditions as keyword arguments
            (e.g., time=15, mechanism="ZEV")

        Example:
            api.set_filter("hackett_2020", time=15, mechanism="ZEV", restriction="P")
            # Equivalent to: WHERE time = 15 AND mechanism = 'ZEV' AND restriction = 'P'

        """
        if not kwargs:
            # If no kwargs provided, clear the filter
            self.clear_filter(config_name)
            return

        # Convert kwargs to SQL WHERE clause
        conditions = []
        for key, value in kwargs.items():
            if isinstance(value, str):
                # String values need quotes
                conditions.append(f"{key} = '{value}'")
            elif value is None:
                # Handle NULL values
                conditions.append(f"{key} IS NULL")
            else:
                # Numeric/boolean values
                conditions.append(f"{key} = {value}")

        where_clause = " AND ".join(conditions)
        self._table_filters[config_name] = where_clause
        self.logger.info(f"Set filter for {config_name}: {where_clause}")

    def set_sql_filter(self, config_name: str, sql_where: str) -> None:
        """
        Set complex filters using SQL WHERE clause.

        Stores raw SQL WHERE clause for automatic application to queries.

        :param config_name: Configuration name to apply filters to
        :param sql_where: SQL WHERE clause (without the 'WHERE' keyword)

        Example:
            api.set_sql_filter("hackett_2020", "time IN (15, 30) AND mechanism = 'ZEV'")

        """
        if not sql_where.strip():
            self.clear_filter(config_name)
            return

        self._table_filters[config_name] = sql_where.strip()
        self.logger.info(f"Set SQL filter for {config_name}: {sql_where}")

    def clear_filter(self, config_name: str) -> None:
        """
        Remove all filters for the specified configuration.

        :param config_name: Configuration name to clear filters for

        """
        if config_name in self._table_filters:
            del self._table_filters[config_name]
            self.logger.info(f"Cleared filter for {config_name}")

    def get_current_filter(self, config_name: str) -> str | None:
        """
        Get the current filter for the specified configuration.

        :param config_name: Configuration name to get filter for
        :return: Current SQL WHERE clause or None if no filter set

        """
        return self._table_filters.get(config_name)

    def query(self, sql: str, config_name: str) -> pd.DataFrame:
        """
        Execute SQL query with automatic filter application.

        Loads the specified configuration, applies any stored filters,
        and executes the query.

        :param sql: SQL query to execute
        :param config_name: Configuration name to query (table will be loaded if needed)
        :return: DataFrame with query results
        :raises ValueError: If config_name not found or query fails

        Example:
            api.set_filter("hackett_2020", time=15, mechanism="ZEV")
            df = api.query("SELECT regulator_locus_tag, target_locus_tag
                FROM hackett_2020", "hackett_2020")
            # Automatically applies: WHERE time = 15 AND mechanism = 'ZEV'

        """
        # Validate config exists
        if config_name not in [c.config_name for c in self.dataset_card.configs]:
            available_configs = [c.config_name for c in self.dataset_card.configs]
            raise ValueError(
                f"Config '{config_name}' not found. "
                f"Available configs: {available_configs}"
            )

        # Load the configuration data
        config = self.dataset_card.get_config_by_name(config_name)
        if not config:
            raise ValueError(f"Could not retrieve config '{config_name}'")

        config_result = self._get_metadata_for_config(config)
        if not config_result.get("success", False):
            raise ValueError(
                f"Failed to load data for config '{config_name}': "
                f"{config_result.get('message', 'Unknown error')}"
            )

        table_name = config_result.get("table_name")
        if not table_name:
            raise ValueError(f"No table available for config '{config_name}'")

        # Replace config name with actual table name in SQL for user convenience
        sql_with_table = sql.replace(config_name, table_name)

        # Apply stored filters
        final_sql = self._apply_filter_to_sql(sql_with_table, config_name)

        try:
            return self.duckdb_conn.execute(final_sql).fetchdf()
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            self.logger.error(f"Final SQL: {final_sql}")
            raise ValueError(f"Query execution failed: {e}") from e

    def _apply_filter_to_sql(self, sql: str, config_name: str) -> str:
        """
        Apply stored filters to SQL query.

        Modifies the SQL query to include stored WHERE clause filters.

        :param sql: Original SQL query
        :param config_name: Configuration name to get filters for
        :return: Modified SQL query with filters applied

        """
        if config_name not in self._table_filters:
            return sql

        filter_clause = self._table_filters[config_name]
        sql_upper = sql.upper()

        if "WHERE" in sql_upper:
            # SQL already has WHERE clause, append with AND
            return f"{sql} AND ({filter_clause})"
        else:
            # Add WHERE clause
            # Find the position to insert WHERE (before ORDER BY, GROUP BY, LIMIT, etc.)
            insert_keywords = ["ORDER BY", "GROUP BY", "HAVING", "LIMIT", "OFFSET"]
            insert_position = len(sql)

            for keyword in insert_keywords:
                pos = sql_upper.find(keyword)
                if pos != -1 and pos < insert_position:
                    insert_position = pos

            if insert_position == len(sql):
                # No special clauses, append WHERE at the end
                return f"{sql} WHERE {filter_clause}"
            else:
                # Insert WHERE before the special clause
                return (
                    f"{sql[:insert_position].rstrip()} "
                    f"WHERE {filter_clause} {sql[insert_position:]}"
                )
