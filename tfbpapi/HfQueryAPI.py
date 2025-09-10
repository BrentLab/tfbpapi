from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import DatasetCard

from .AbstractHfAPI import AbstractHfAPI


class HfQueryAPI(AbstractHfAPI):
    """
    Concrete implementation of AbstractHfAPI with DuckDB query capabilities.

    This class provides seamless querying of Hugging Face datasets using SQL via DuckDB.
    It automatically handles dataset downloading, parsing, and provides a simple query
    interface.

    """

    def __init__(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
        token: str | None = None,
        cache_dir: str | Path | None = None,
        auto_download_threshold_mb: float = 100.0,
        auto_parse_datacard: bool = True,
    ):
        """
        Initialize the HF Query API client.

        :param repo_id: The repo identifier on HF (e.g., "user/dataset")
        :param repo_type: One of {"model", "dataset", "space"}. Defaults to "dataset"
        :param token: Optional HF token for private repos
        :param cache_dir: HF cache directory for downloads
        :param auto_download_threshold_mb: Auto-download threshold in MB
        :param auto_parse_datacard: Whether to automatically parse datacard on init

        """
        super().__init__(repo_id, repo_type, token, cache_dir)
        self.auto_download_threshold_mb = auto_download_threshold_mb
        self._datasets: dict[str, Any] = {}
        self._loaded_datasets: dict[str, Dataset | DatasetDict] = {}
        self._duckdb_conn = duckdb.connect(":memory:")
        self._table_filters: dict[str, str] = {}

        if auto_parse_datacard:
            try:
                self.datasets = self.parse_datacard()
            except Exception as e:
                self.logger.warning(f"Failed to auto-parse datacard: {e}")
                self._datasets = {}

    @property
    def datasets(self) -> dict[str, Any]:
        """Parsed dataset configurations from the datacard."""
        return self._datasets

    @datasets.setter
    def datasets(self, value: dict[str, Any]) -> None:
        """Set the parsed datasets and update available tables."""
        self._datasets = value
        self._update_available_tables()

    @property
    def available_tables(self) -> list[str]:
        """List of available table names for querying."""
        return list(self._datasets.keys())

    def parse_datacard(self) -> dict[str, Any]:
        """
        Parse the dataset card into a standardized format.

        :return: Dict mapping config names to their metadata

        """
        try:
            card = DatasetCard.load(self.repo_id, self.repo_type)
            data_dict = card.data.to_dict()
        except Exception as e:
            self.logger.error(f"Failed to load dataset card: {e}")
            return {}

        parsed_datasets = {}

        for config in data_dict.get("configs", []):
            config_name = config["config_name"]

            # Extract features for filtering/querying
            features = {}
            if "dataset_info" in config and "features" in config["dataset_info"]:
                for feature in config["dataset_info"]["features"]:
                    features[feature["name"]] = {
                        "dtype": feature["dtype"],
                        "description": feature.get("description", ""),
                    }

            # Extract file paths
            data_files = []
            for file_info in config.get("data_files", []):
                data_files.append(
                    {
                        "path": file_info["path"],
                        "split": file_info.get("split", "train"),
                    }
                )

            parsed_datasets[config_name] = {
                "features": features,
                "data_files": data_files,
                "config": config,
                "loaded": False,
            }

        return parsed_datasets

    def _update_available_tables(self) -> None:
        """Update the logger with information about available tables."""
        if self._datasets:
            self.logger.info(f"Available tables: {', '.join(self.available_tables)}")

    def _ensure_dataset_loaded(self, table_name: str) -> Dataset | DatasetDict:
        """
        Ensure a dataset is loaded and available for querying.

        :param table_name: Name of the dataset configuration
        :return: The loaded dataset
        :raises ValueError: If table_name is not found

        """
        if table_name not in self._datasets:
            raise ValueError(
                f"Table '{table_name}' not found. "
                f"Available tables: {self.available_tables}"
            )

        if table_name in self._loaded_datasets:
            return self._loaded_datasets[table_name]

        # Download the dataset if not already downloaded
        if not self.snapshot_path:
            self.logger.info(f"Downloading dataset for table '{table_name}'...")
            self.download(auto_download_threshold_mb=self.auto_download_threshold_mb)

        # Load the specific dataset configuration
        try:
            self.logger.info(f"Loading dataset configuration '{table_name}'...")
            dataset = load_dataset(
                str(self.snapshot_path), name=table_name, keep_in_memory=False
            )
            self._loaded_datasets[table_name] = dataset
            self._datasets[table_name]["loaded"] = True

            # Register with DuckDB
            self._register_dataset_with_duckdb(table_name, dataset)

            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset '{table_name}': {e}")
            raise

    def _register_dataset_with_duckdb(
        self, table_name: str, dataset: Dataset | DatasetDict
    ) -> None:
        """Register a dataset with DuckDB for SQL querying."""
        try:
            if isinstance(dataset, DatasetDict):
                # Register each split as a separate view
                for split_name, split_dataset in dataset.items():
                    view_name = (
                        f"{table_name}_{split_name}"
                        if split_name != "train"
                        else table_name
                    )
                    df = split_dataset.to_pandas()
                    self._duckdb_conn.register(view_name, df)
                    self.logger.debug(
                        f"Registered view '{view_name}' with {len(df)} rows"
                    )
            else:
                # Single dataset
                df = dataset.to_pandas()
                self._duckdb_conn.register(table_name, df)
                self.logger.debug(
                    f"Registered table '{table_name}' with {len(df)} rows"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to register dataset '{table_name}' with DuckDB: {e}"
            )
            raise

    def query(self, sql: str, table_name: str | None = None) -> pd.DataFrame:
        """
        Execute a SQL query against the dataset.

        :param sql: SQL query string
        :param table_name: Optional table name to ensure is loaded. If not provided,
            attempts to infer from the SQL query
        :return: Query results as a pandas DataFrame

        """
        # If table_name not provided, try to infer from available tables
        if table_name is None:
            sql_lower = sql.lower()
            for available_table in self.available_tables:
                if available_table in sql_lower:
                    table_name = available_table
                    break

        # If we found a table name, ensure it's loaded
        if table_name:
            self._ensure_dataset_loaded(table_name)
        elif not self._loaded_datasets:
            # If no datasets are loaded and we couldn't infer,
            # try to load the first one
            if self.available_tables:
                self._ensure_dataset_loaded(self.available_tables[0])

        # Apply any table filters to the query
        modified_sql = self._apply_table_filters(sql)

        try:
            result = self._duckdb_conn.execute(modified_sql).fetchdf()
            self.logger.debug(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            if modified_sql != sql:
                self.logger.debug(f"Original query: {sql}")
                self.logger.debug(f"Modified query: {modified_sql}")
            raise

    def describe_table(self, table_name: str) -> pd.DataFrame:
        """
        Get information about a table's structure.

        :param table_name: Name of the table to describe
        :return: DataFrame with column information

        """
        self._ensure_dataset_loaded(table_name)
        return self.query(f"DESCRIBE {table_name}")

    def sample(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of rows from a table.

        :param table_name: Name of the table to sample
        :param n: Number of rows to return
        :return: Sample DataFrame

        """
        return self.query(f"SELECT * FROM {table_name} LIMIT {n}", table_name)

    def count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.

        :param table_name: Name of the table to count
        :return: Number of rows

        """
        result = self.query(f"SELECT COUNT(*) as count FROM {table_name}", table_name)
        return result.iloc[0]["count"]

    def get_columns(self, table_name: str) -> list[str]:
        """
        Get column names for a table.

        :param table_name: Name of the table
        :return: List of column names

        """
        if table_name not in self._datasets:
            raise ValueError(f"Table '{table_name}' not found")

        return list(self._datasets[table_name]["features"].keys())

    def _apply_table_filters(self, sql: str) -> str:
        """
        Apply table filters to a SQL query by modifying table references.

        :param sql: Original SQL query
        :return: Modified SQL query with filters applied

        """
        if not self._table_filters:
            return sql

        modified_sql = sql

        # Apply filters by replacing table references with filtered subqueries
        for table_name, filter_condition in self._table_filters.items():
            import re

            # Simple pattern to match table references in FROM and JOIN clauses
            pattern = rf"\b{re.escape(table_name)}\b"
            replacement = f"(SELECT * FROM {table_name} WHERE {filter_condition})"

            # Only replace if we find the table name in the SQL
            if re.search(pattern, modified_sql, re.IGNORECASE):
                # Check if it's already wrapped in a filtered subquery to
                # avoid double-wrapping
                if not re.search(
                    rf"\(SELECT.*FROM\s+{re.escape(table_name)}\s+WHERE",
                    modified_sql,
                    re.IGNORECASE,
                ):
                    modified_sql = re.sub(
                        pattern, replacement, modified_sql, flags=re.IGNORECASE
                    )

        return modified_sql

    def get_table_filter(self, table_name: str) -> str | None:
        """
        Get the current filter for a table.

        :param table_name: Name of the table
        :return: Current filter SQL condition or None if no filter is set

        """
        return self._table_filters.get(table_name)

    def set_table_filter(self, table_name: str, filter_condition: str) -> None:
        """
        Set a filter condition for a table that will be applied to all queries.

        :param table_name: Name of the table
        :param filter_condition: SQL WHERE condition (without the WHERE keyword)

        """
        self._table_filters[table_name] = filter_condition
        self.logger.debug(f"Set filter for table '{table_name}': {filter_condition}")

    def remove_table_filter(self, table_name: str) -> None:
        """
        Remove any filter condition for a table.

        :param table_name: Name of the table

        """
        removed_filter = self._table_filters.pop(table_name, None)
        if removed_filter:
            self.logger.debug(
                f"Removed filter for table '{table_name}': {removed_filter}"
            )

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._duckdb_conn:
            self._duckdb_conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
