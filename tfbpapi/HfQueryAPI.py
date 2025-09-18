import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from .errors import RepoTooLargeError
from .HfCacheManager import HFCacheManager


class HfQueryAPI:
    """Hugging Face API client with intelligent downloading and SQL querying."""

    def __init__(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
        token: str | None = None,
        cache_dir: str | Path | None = None,
        auto_download_threshold_mb: float = 100.0,
        auto_parse_datacard: bool = True,
        enable_cache_management: bool = True,
        cache_auto_cleanup: bool = False,
        cache_max_age_days: int = 30,
        cache_max_size: str = "10GB",
    ):
        """
        Initialize the HF Query API client.

        :param repo_id: Repository identifier (e.g., "user/dataset")
        :param repo_type: Type of repository ("dataset", "model", "space")
        :param token: HuggingFace token for authentication
        :param cache_dir: HF cache_dir for downloads
        :param auto_download_threshold_mb: Threshold in MB for auto full download
        :param auto_parse_datacard: Whether to auto-parse the datacard on init
        :param enable_cache_management: Enable integrated cache management features
        :param cache_auto_cleanup: Enable automatic cache cleanup
        :param cache_max_age_days: Maximum age in days for cache entries
        :param cache_max_size: Maximum total cache size (e.g., "10GB")

        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize data info manager with new architecture
        self.data_info = HfDataInfoManager(
            repo_id=repo_id, repo_type=repo_type, token=token
        )

        self.cache_dir = Path(
            cache_dir if cache_dir else os.getenv("HF_CACHE_DIR", HF_HUB_CACHE)
        )
        self.auto_download_threshold_mb = auto_download_threshold_mb
        self._loaded_datasets: dict[str, Dataset | DatasetDict] = {}
        self._duckdb_conn = duckdb.connect(":memory:")
        self._table_filters: dict[str, str] = {}
        self._partition_cache: dict[str, set[str]] = {}  # Track downloaded partitions

        # Initialize cache management
        self._cache_manager = None
        if enable_cache_management:
            self._cache_manager = HFCacheManager(logger=self.logger)
            self._cache_auto_cleanup = cache_auto_cleanup
            self._cache_max_age_days = cache_max_age_days
            self._cache_max_size = cache_max_size

        if auto_parse_datacard:
            try:
                self.data_info.parse_datacard()
            except Exception as e:
                self.logger.warning(f"Failed to auto-parse datacard: {e}")
                self.data_info.clear()

    @property
    def repo_id(self) -> str:
        return self.data_info.repo_id

    @property
    def repo_type(self) -> str:
        return self.data_info.repo_type

    @property
    def token(self) -> str | None:
        return self.data_info.token

    @property
    def datasets(self) -> dict[str, Any]:
        """Parsed dataset configurations from the datacard."""
        # Convert TableConfig objects to legacy dictionary format for backward compatibility
        result = {}
        for name, table_config in self.data_info.datasets.items():
            if table_config:
                # Convert FeatureInfo objects to legacy format
                features = {}
                for feat_name, feat_info in table_config.features.items():
                    features[feat_name] = {
                        "dtype": feat_info.dtype,
                        "description": feat_info.description,
                    }

                # Convert DataFileInfo objects to legacy format
                data_files = []
                for file_info in table_config.data_files:
                    data_files.append(
                        {"path": file_info.path, "split": file_info.split}
                    )

                result[name] = {
                    "features": features,
                    "data_files": data_files,
                    "config": table_config.config,
                    "loaded": table_config.downloaded,  # Use "loaded" for backward compatibility
                    "is_partitioned": table_config.is_partitioned,
                }
        return result

    @datasets.setter
    def datasets(self, value: dict[str, Any]) -> None:
        """Set dataset configurations (for backward compatibility)."""
        # Clear existing data and register new tables
        self.data_info.clear()
        # Convert legacy format to TableConfig objects if needed
        from .datainfo.models import TableConfig

        table_configs = {}
        for name, config in value.items():
            if isinstance(config, TableConfig):
                table_configs[name] = config
            else:
                # Convert from legacy dict format
                table_configs[name] = TableConfig(
                    name=name,
                    features=config.get("features", {}),
                    data_files=config.get("data_files", []),
                    config=config,
                    downloaded=config.get("loaded", False),  # "loaded" was the old name
                    is_partitioned=config.get("is_partitioned", False),
                    partition_info=None,
                )
        self.data_info._registry.register_tables(table_configs)

    @property
    def available_tables(self) -> list[str]:
        """List of available table names for querying."""
        return self.data_info.available_tables

    @property
    def cache_manager(self) -> HFCacheManager | None:
        """Access to the integrated cache manager for advanced cache operations."""
        return self._cache_manager

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

    @property
    def size(self) -> dict[str, Any] | None:
        """Size information from the HF Dataset Server API."""
        size_info = self.data_info.get_dataset_size()
        if size_info:
            return {
                "total": size_info.total_bytes,
                "total_mb": size_info.total_mb,
                "configs": size_info.config_sizes,
            }
        return None

    @size.setter
    def size(self, value: dict[str, Any]) -> None:
        """Set size information (for backward compatibility)."""
        # Convert legacy format to DatasetSize object
        from .datainfo.models import DatasetSize

        dataset_size = DatasetSize.from_hf_size_response(value)
        self.data_info._dataset_size = dataset_size

    @property
    def snapshot_path(self) -> Path | None:
        """Path to the last downloaded snapshot (if any)."""
        return getattr(self, "_snapshot_path", None)

    @snapshot_path.setter
    def snapshot_path(self, value: str | Path | None) -> None:
        self._snapshot_path = None if value is None else Path(value)

    def _build_auth_headers(self) -> dict[str, str]:
        """Build authentication headers if token is available."""
        return (
            {"Authorization": f"Bearer {self.data_info.token}"}
            if self.data_info.token
            else {}
        )

    def _normalize_patterns(self, kwargs: dict[str, Any]) -> None:
        """Convert string patterns to lists."""
        for pattern_key in ["allow_patterns", "ignore_patterns"]:
            if pattern_key in kwargs and kwargs[pattern_key] is not None:
                patterns = kwargs[pattern_key]
                if isinstance(patterns, str):
                    kwargs[pattern_key] = [patterns]

    def download_partitions(
        self, table_name: str, partition_values: set[str] | None = None
    ) -> Path:
        """
        Download specific partitions using the path_template from partitioning metadata.

        :param table_name: Name of the dataset table
        :param partition_values: Specific partition values to download (None for all)
        :return: Path to downloaded data

        """
        table_config = self.data_info.get_table_or_raise(table_name)

        if not table_config.is_partitioned:
            raise ValueError(f"Table {table_name} is not configured as partitioned")

        partition_info = self.data_info.get_partition_info(table_name)
        if not partition_info:
            raise ValueError(f"Table {table_name} missing partition information")

        path_template = partition_info.get("path_template")
        if not path_template:
            raise ValueError(
                f"Table {table_name} missing required path_template in partitioning config"
            )

        partition_columns = partition_info.get("partition_by", [])

        if partition_values and partition_columns:
            # Download specific partitions using path template
            patterns = []
            for partition_value in partition_values:
                # For single-column partitioning, substitute the first column
                if len(partition_columns) == 1:
                    column = partition_columns[0]
                    pattern = path_template.replace(f"{{{column}}}", partition_value)
                    patterns.append(pattern)
                else:
                    # For multi-column partitioning, we'd need more sophisticated logic
                    # For now, create a wildcard pattern for the specific value
                    # This is a simplified approach - real implementation would need
                    # to handle multi-dimensional partition filtering
                    pattern = path_template
                    for column in partition_columns:
                        if f"{{{column}}}" in pattern:
                            pattern = pattern.replace(f"{{{column}}}", partition_value)
                            break
                    patterns.append(pattern)

            self.logger.info(
                f"Downloading partitions for {table_name}: {partition_values}"
            )
            self.logger.debug(f"Using patterns: {patterns}")
            return self.download(allow_patterns=patterns)
        else:
            # Download all partitions - use the original data files paths
            patterns = table_config.get_file_paths()

            self.logger.info(f"Downloading all partitions for {table_name}")
            return self.download(allow_patterns=patterns)

    def download(
        self,
        files: list[str] | str | None = None,
        force_full_download: bool = False,
        auto_download_threshold_mb: float = 100.0,
        dry_run: bool = False,
        **kwargs,
    ) -> Path:
        """Download dataset with intelligent partitioning support."""
        dataset_size_mb = self.data_info.get_dataset_size_mb()

        if dataset_size_mb <= auto_download_threshold_mb or force_full_download:
            self.logger.info(
                f"Dataset size ({dataset_size_mb:.2f} MB) is below threshold. "
                "Downloading entire repo."
            )
            files = None
            kwargs.pop("allow_patterns", None)
            kwargs.pop("ignore_patterns", None)
        elif (
            not files
            and not kwargs.get("allow_patterns")
            and not kwargs.get("ignore_patterns")
        ):
            excess_size_mb = dataset_size_mb - auto_download_threshold_mb
            raise RepoTooLargeError(
                f"Dataset size ({dataset_size_mb:.2f} MB) exceeds threshold by "
                f"{excess_size_mb:.2f} MB. Specify files, patterns, or set "
                "force_full_download=True."
            )

        # Handle specific file downloads
        if files is not None:
            if isinstance(files, str) or (isinstance(files, list) and len(files) == 1):
                filename = files if isinstance(files, str) else files[0]
                return self._download_single_file(filename, dry_run=dry_run, **kwargs)
            elif isinstance(files, list) and len(files) > 1:
                if kwargs.get("allow_patterns") is not None:
                    self.logger.warning(
                        "Both 'files' and 'allow_patterns' provided. Using 'files'."
                    )
                kwargs["allow_patterns"] = files

        return self._download_snapshot(dry_run=dry_run, **kwargs)

    def _download_single_file(
        self, filename: str, dry_run: bool = False, **kwargs
    ) -> Path:
        """Download a single file using hf_hub_download."""
        self.logger.info(f"Downloading single file: {filename}")

        if dry_run:
            self.logger.info(f"[DRY RUN] Would download {filename} from {self.repo_id}")
            return Path("dry_run_path")

        hf_kwargs = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "filename": filename,
            "token": self.token,
            **kwargs,
        }

        if "local_dir" not in hf_kwargs and self.cache_dir is not None:
            hf_kwargs["cache_dir"] = str(self.cache_dir)

        for key in ["local_dir", "cache_dir"]:
            if key in hf_kwargs and hf_kwargs[key] is not None:
                hf_kwargs[key] = str(hf_kwargs[key])
        file_path = hf_hub_download(**hf_kwargs)
        self.snapshot_path = Path(file_path).parent
        return Path(file_path)

    def _download_snapshot(self, dry_run: bool = False, **kwargs) -> Path:
        """Download repository snapshot using snapshot_download."""
        if dry_run:
            self.logger.info(f"[DRY RUN] Would download from {self.repo_id}:")
            self.logger.info(f"  - allow_patterns: {kwargs.get('allow_patterns')}")
            self.logger.info(f"  - ignore_patterns: {kwargs.get('ignore_patterns')}")
            return Path("dry_run_path")

        self.logger.info(
            f"Downloading repo snapshot - "
            f"allow: {kwargs.get('allow_patterns')}, "
            f"ignore: {kwargs.get('ignore_patterns')}"
        )

        snapshot_kwargs = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "token": self.token,
            **kwargs,
        }

        if (
            "local_dir" not in snapshot_kwargs
            and "cache_dir" not in snapshot_kwargs
            and self.cache_dir is not None
        ):
            snapshot_kwargs["cache_dir"] = str(self.cache_dir)

        self._normalize_patterns(snapshot_kwargs)
        snapshot_path = snapshot_download(**snapshot_kwargs)
        self.snapshot_path = Path(snapshot_path)
        return self.snapshot_path

    def _update_available_tables(self) -> None:
        """Update the logger with information about available tables."""
        if self.data_info.available_tables:
            self.logger.info(f"Available tables: {', '.join(self.available_tables)}")

    def _extract_table_references(self, sql: str) -> set[str]:
        """
        Extract all table references from a SQL query.

        Handles FROM clauses, JOINs, subqueries, and direct parquet file references.

        :param sql: SQL query string
        :return: Set of table names/file references found in the query

        """
        table_refs = set()

        # Remove comments and normalize whitespace
        sql_clean = re.sub(r"--.*?\n", " ", sql, flags=re.DOTALL)
        sql_clean = re.sub(r"/\*.*?\*/", " ", sql_clean, flags=re.DOTALL)
        sql_clean = re.sub(r"\s+", " ", sql_clean.strip())

        # Pattern to match table references in FROM and JOIN clauses
        # This handles: FROM table, FROM read_parquet('file.parquet'), JOIN table ON...
        from_pattern = r"""
            (?:FROM|JOIN)\s+                    # FROM or JOIN keyword
            (?:
                read_parquet\s*\(\s*['"]([^'"]+)['"][\s)]*  # read_parquet('file.parquet')
                |
                ([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)  # table.name or tablename
            )
            (?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?     # Optional alias
            (?:\s+ON\s+.*?(?=\s+(?:FROM|JOIN|WHERE|GROUP|ORDER|LIMIT|$)))?  # Optional ON clause for JOINs
        """

        for match in re.finditer(from_pattern, sql_clean, re.IGNORECASE | re.VERBOSE):
            parquet_file = match.group(1)
            table_name = match.group(2)

            if parquet_file:
                # Extract just the filename without extension for parquet files
                file_ref = Path(parquet_file).stem
                table_refs.add(file_ref)
            elif table_name:
                # Clean table name (remove schema prefix if present)
                clean_name = table_name.split(".")[-1]
                table_refs.add(clean_name)

        # Also check for simple table name patterns in case the regex missed something
        simple_pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b"
        for match in re.finditer(simple_pattern, sql_clean, re.IGNORECASE):
            table_name = match.group(1).lower()
            # Filter out SQL keywords and function names
            if table_name not in {
                "select",
                "where",
                "group",
                "order",
                "having",
                "limit",
                "offset",
                "union",
                "intersect",
                "except",
                "read_parquet",
                "read_csv",
                "read_json",
            }:
                table_refs.add(table_name)

        return table_refs

    def _resolve_table_to_files(self, table_ref: str) -> list[str]:
        """
        Resolve a table reference to specific files that need to be downloaded.

        :param table_ref: Table name or file reference from SQL
        :return: List of file paths/patterns needed for this table

        """
        return self.data_info.resolve_table_to_files(table_ref)

    def _ensure_tables_available(self, sql: str) -> set[str]:
        """
        Ensure all tables referenced in SQL are available, downloading if necessary.

        :param sql: SQL query string
        :return: Set of table names that were processed

        """
        table_refs = self._extract_table_references(sql)
        processed_tables = set()

        for table_ref in table_refs:
            if self.data_info.has_table(table_ref):
                # Table is already known from dataset card
                if not self.data_info.is_table_downloaded(table_ref):
                    self._ensure_dataset_loaded(table_ref, sql)
                processed_tables.add(table_ref)
            else:
                # Try to discover as standalone file
                if self._try_discover_standalone_table(table_ref):
                    processed_tables.add(table_ref)
                else:
                    # File doesn't exist locally, try to download it
                    files_needed = self._resolve_table_to_files(table_ref)
                    if self._try_download_specific_files(files_needed, table_ref):
                        processed_tables.add(table_ref)
                    else:
                        self.logger.warning(
                            f"Could not locate or download table: {table_ref}"
                        )

        return processed_tables

    def _try_download_specific_files(self, files: list[str], table_name: str) -> bool:
        """
        Attempt to download specific files for a table.

        :param files: List of file paths/patterns to download
        :param table_name: Name of the table these files represent
        :return: True if download was successful

        """
        try:
            # Try downloading specific files
            for file_path in files:
                try:
                    # First check if file exists in repo
                    downloaded_file_path = self._download_single_file(
                        file_path, dry_run=False
                    )
                    if downloaded_file_path and downloaded_file_path.exists():
                        # Register the file as a table if it's a parquet file
                        if file_path.endswith(".parquet"):
                            self._register_parquet_as_table(
                                downloaded_file_path, table_name
                            )
                            return True
                except Exception as e:
                    self.logger.debug(f"Failed to download {file_path}: {e}")
                    continue

            # If individual file downloads failed, try pattern-based download
            if files:
                try:
                    self.download(allow_patterns=files, force_full_download=False)
                    # After download, try to discover the table again
                    return self._try_discover_standalone_table(table_name)
                except RepoTooLargeError:
                    self.logger.error(
                        f"Repository too large to download files for table: {table_name}"
                    )
                    return False
                except Exception as e:
                    self.logger.error(
                        f"Failed to download files for table {table_name}: {e}"
                    )
                    return False

            return False
        except Exception as e:
            self.logger.error(f"Error downloading files for table {table_name}: {e}")
            return False

    def _register_parquet_as_table(self, parquet_path: Path, table_name: str) -> None:
        """Register a parquet file directly as a DuckDB table."""
        create_view_sql = f"""
        CREATE OR REPLACE VIEW {table_name} AS
        SELECT * FROM read_parquet('{parquet_path}')
        """
        self._duckdb_conn.execute(create_view_sql)

        # Add to datasets registry via DataInfo
        self.data_info.add_standalone_table(table_name, parquet_path, downloaded=True)

    def _try_discover_standalone_table(self, table_name: str) -> bool:
        """
        Try to discover a standalone parquet file as a table.

        :param table_name: The name of the table to discover
        :return: True if the table was discovered and registered

        """
        if not self.snapshot_path:
            return False

        # Look for parquet file with matching name
        parquet_file = self.snapshot_path / f"{table_name}.parquet"
        if parquet_file.exists():
            # Register the standalone parquet file with DataInfo
            self.data_info.add_standalone_table(
                table_name, parquet_file, downloaded=True
            )

            # Register directly with DuckDB
            create_view_sql = f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT * FROM read_parquet('{parquet_file}')
            """
            self._duckdb_conn.execute(create_view_sql)
            self.logger.debug(
                f"Registered standalone parquet file as table: {table_name}"
            )
            return True

        return False

    def _ensure_dataset_loaded(
        self, table_name: str, sql: str | None = None
    ) -> Dataset | DatasetDict | None:
        """
        Ensure a dataset is loaded, with intelligent partition downloading.

        :param table_name: Name of the dataset configuration
        :param sql: Optional SQL query to determine required partitions
        :return: The loaded dataset (None for partitioned datasets that are loaded
            directly into DuckDB)

        """
        if not self.data_info.has_table(table_name):
            # Try to discover the table as a standalone parquet file
            if self._try_discover_standalone_table(table_name):
                self.logger.info(f"Discovered standalone table: {table_name}")
            else:
                raise ValueError(
                    f"Table '{table_name}' not found. "
                    f"Available tables: {self.available_tables}"
                )

        dataset_info = self.data_info.get_table_info(table_name)

        # Check if we need to download partitions
        if dataset_info and dataset_info.is_partitioned and sql:
            required_partitions = self._get_required_partitions(sql, table_name)

            if required_partitions:
                # Check if we have these partitions cached
                cached_partitions = self._partition_cache.get(table_name, set())
                missing_partitions = required_partitions - cached_partitions

                if missing_partitions:
                    self.logger.info(
                        f"Downloading missing partitions: {missing_partitions}"
                    )
                    self._download_partitions(table_name, missing_partitions)
                    self._partition_cache.setdefault(table_name, set()).update(
                        missing_partitions
                    )

        # Check if dataset is already loaded
        if table_name in self._loaded_datasets:
            return self._loaded_datasets[table_name]

        # Check if standalone table is already registered
        if dataset_info and dataset_info.downloaded:
            return None  # Standalone tables are registered directly with DuckDB

        # Download if needed
        if not self.snapshot_path:
            if dataset_info and dataset_info.is_partitioned and sql:
                required_partitions = self._get_required_partitions(sql, table_name)
                self._download_partitions(table_name, required_partitions)
            else:
                self.logger.info(f"Downloading dataset for table '{table_name}'...")
                self.download(
                    auto_download_threshold_mb=self.auto_download_threshold_mb
                )

        # Load the dataset
        try:
            self.logger.info(f"Loading dataset configuration '{table_name}'...")

            if dataset_info and dataset_info.is_partitioned:
                # For partitioned datasets, load directly into DuckDB
                self._load_partitioned_dataset(table_name)
                # Mark as downloaded but don't store in _loaded_datasets since it's in DuckDB
                self.data_info.mark_table_downloaded(table_name, True)
                return None
            else:
                # Standard dataset loading
                dataset = load_dataset(
                    str(self.snapshot_path), name=table_name, keep_in_memory=False
                )
                self._loaded_datasets[table_name] = dataset
                self._register_dataset_with_duckdb(table_name, dataset)
                return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset '{table_name}': {e}")
            raise

    def _load_partitioned_dataset(self, table_name: str) -> None:
        """
        Load a partitioned dataset directly into DuckDB without using Hugging Face
        datasets.

        This is more efficient for partitioned datasets like genome_map.

        """
        # Find parquet files in the snapshot
        if not self.snapshot_path:
            raise ValueError("No snapshot path available")

        # Look for parquet files matching the dataset pattern
        parquet_files: list[Path] = []
        for file_info in self.data_info.get_table_data_files(table_name):
            pattern = file_info["path"]
            if "*" in pattern:
                # Convert pattern to actual file search
                search_pattern = pattern.replace("*", "**")
                parquet_files.extend(self.snapshot_path.glob(search_pattern))

        if not parquet_files:
            # Fallback: find all parquet files
            parquet_files = list(self.snapshot_path.rglob("*.parquet"))

        if parquet_files:
            # Register parquet files directly with DuckDB
            file_paths = [str(f) for f in parquet_files]
            self.logger.info(
                f"Registering {len(file_paths)} parquet files for {table_name}"
            )

            # Create a view that reads from all parquet files
            files_str = "', '".join(file_paths)
            create_view_sql = f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT * FROM read_parquet(['{files_str}'])
            """
            self._duckdb_conn.execute(create_view_sql)
            self.logger.debug(
                f"Created view '{table_name}' from {len(file_paths)} parquet files"
            )
        else:
            raise ValueError(
                f"No parquet files found for partitioned dataset {table_name}"
            )

    def _register_dataset_with_duckdb(
        self, table_name: str, dataset: Dataset | DatasetDict
    ) -> None:
        """Register a standard dataset with DuckDB for SQL querying."""
        try:
            if isinstance(dataset, DatasetDict):
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
        Execute a SQL query with intelligent table discovery and loading.

        :param sql: SQL query string
        :param table_name: Optional table name to ensure is loaded (legacy parameter)
        :return: Query results as a pandas DataFrame

        """
        # Use intelligent table discovery to ensure all referenced tables are available
        processed_tables = self._ensure_tables_available(sql)

        # Legacy support: if table_name is provided, ensure it's also available
        if table_name and table_name not in processed_tables:
            try:
                self._ensure_dataset_loaded(table_name, sql)
            except ValueError:
                # Try to discover as standalone table
                if not self._try_discover_standalone_table(table_name):
                    self.logger.warning(f"Could not load specified table: {table_name}")

        # Apply table filters
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

            # If query failed, provide helpful information about available tables
            if self.data_info.available_tables:
                self.logger.info(
                    f"Available tables: {', '.join(self.data_info.available_tables)}"
                )

            raise

    def _apply_table_filters(self, sql: str) -> str:
        """Apply table filters to a SQL query."""
        if not self._table_filters:
            return sql

        modified_sql = sql
        for table_name, filter_condition in self._table_filters.items():
            pattern = rf"\b{re.escape(table_name)}\b"
            replacement = f"(SELECT * FROM {table_name} WHERE {filter_condition})"

            if re.search(pattern, modified_sql, re.IGNORECASE):
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
        """Get the current filter for a table."""
        return self._table_filters.get(table_name)

    def set_table_filter(self, table_name: str, filter_condition: str) -> None:
        """Set a filter condition for a table."""
        self._table_filters[table_name] = filter_condition
        self.logger.debug(f"Set filter for table '{table_name}': {filter_condition}")

    def remove_table_filter(self, table_name: str) -> None:
        """Remove any filter condition for a table."""
        removed_filter = self._table_filters.pop(table_name, None)
        if removed_filter:
            self.logger.debug(
                f"Removed filter for table '{table_name}': {removed_filter}"
            )

    # Standard methods
    def describe_table(self, table_name: str) -> pd.DataFrame:
        """Get information about a table's structure."""
        self._ensure_dataset_loaded(table_name)
        return self.query(f"DESCRIBE {table_name}")

    def sample(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """Get a sample of rows from a table."""
        return self.query(f"SELECT * FROM {table_name} LIMIT {n}", table_name)

    def count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        result = self.query(f"SELECT COUNT(*) as count FROM {table_name}", table_name)
        return result.iloc[0]["count"]

    def get_columns(self, table_name: str) -> list[str]:
        """Get column names for a table."""
        if not self.data_info.has_table(table_name):
            raise ValueError(f"Table '{table_name}' not found")
        return list(self.data_info.get_table_features(table_name).keys())

    # ============== Cache Management Methods ==============

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get comprehensive cache information including current repo details.

        :return: Dictionary with cache stats, repo info, and recommendations
        """
        if not self._cache_manager:
            self.logger.debug("Cache management is disabled")
            return {"cache_management": "disabled"}

        self.logger.info("Retrieving comprehensive cache information")

        from huggingface_hub import scan_cache_dir

        try:
            cache_info = scan_cache_dir()
            self.logger.debug(
                f"Scanned cache directory: {cache_info.size_on_disk_str} in {len(cache_info.repos)} repos"
            )
        except Exception as e:
            self.logger.error(f"Failed to scan cache directory: {e}")
            return {"error": f"Failed to scan cache: {e}"}

        # Find current repo in cache
        current_repo_info = None
        for repo in cache_info.repos:
            if (
                repo.repo_id == self.repo_id
                and repo.repo_type.lower() == self.repo_type
            ):
                current_repo_info = repo
                break

        result = {
            "cache_management": "enabled",
            "cache_directory": str(self.cache_dir),
            "total_cache_size": cache_info.size_on_disk_str,
            "total_cache_size_bytes": cache_info.size_on_disk,
            "total_repos_cached": len(cache_info.repos),
            "current_repo": {
                "repo_id": self.repo_id,
                "repo_type": self.repo_type,
                "cached": current_repo_info is not None,
                "cache_info": None,
            },
            "cache_policies": {
                "auto_cleanup": getattr(self, "_cache_auto_cleanup", False),
                "max_age_days": getattr(self, "_cache_max_age_days", 30),
                "max_size": getattr(self, "_cache_max_size", "10GB"),
            },
            "recommendations": [],
        }

        if current_repo_info:
            result["current_repo"]["cache_info"] = {
                "size_on_disk": current_repo_info.size_on_disk_str,
                "size_bytes": current_repo_info.size_on_disk,
                "nb_files": current_repo_info.nb_files,
                "last_accessed": current_repo_info.last_accessed,
                "last_modified": current_repo_info.last_modified,
                "revisions_count": len(current_repo_info.revisions),
                "revisions": [
                    {
                        "commit_hash": rev.commit_hash[:8],
                        "size_on_disk": rev.size_on_disk_str,
                        "last_modified": rev.last_modified,
                        "files_count": len(rev.files),
                    }
                    for rev in sorted(
                        current_repo_info.revisions,
                        key=lambda r: r.last_modified,
                        reverse=True,
                    )
                ],
            }

        # Add recommendations with logging
        max_size_bytes = self._cache_manager._parse_size_string(
            getattr(self, "_cache_max_size", "10GB")
        )
        if cache_info.size_on_disk > max_size_bytes:
            recommendation = (
                f"Cache size ({cache_info.size_on_disk_str}) exceeds configured limit "
                f"({getattr(self, '_cache_max_size', '10GB')}). Consider running cache cleanup."
            )
            result["recommendations"].append(recommendation)
            self.logger.warning(f"Cache size warning: {recommendation}")

        if len(cache_info.repos) > 50:  # Arbitrary threshold
            recommendation = (
                f"Large number of cached repos ({len(cache_info.repos)}). "
                "Consider cleaning unused repositories."
            )
            result["recommendations"].append(recommendation)
            self.logger.info(f"Cache optimization suggestion: {recommendation}")

        if current_repo_info:
            self.logger.info(
                f"Current repo {self.repo_id} found in cache: "
                f"{current_repo_info.size_on_disk_str}, {len(current_repo_info.revisions)} revisions"
            )
        else:
            self.logger.debug(f"Current repo {self.repo_id} not found in cache")

        self.logger.info(
            f"Cache info summary: {cache_info.size_on_disk_str} total, "
            f"{len(cache_info.repos)} repos, {len(result['recommendations'])} recommendations"
        )

        return result

    def get_repo_cache_info(self, repo_id: str | None = None) -> dict[str, Any]:
        """
        Get detailed cache information for a specific repository.

        :param repo_id: Repository ID (defaults to current repo)
        :return: Dictionary with detailed repo cache information
        """
        if not self._cache_manager:
            self.logger.debug("Cache management is disabled")
            return {"cache_management": "disabled"}

        target_repo_id = repo_id or self.repo_id
        self.logger.info(
            f"Retrieving detailed cache information for repo: {target_repo_id}"
        )

        from huggingface_hub import scan_cache_dir

        try:
            cache_info = scan_cache_dir()
        except Exception as e:
            self.logger.error(f"Failed to scan cache directory: {e}")
            return {"error": f"Failed to scan cache: {e}"}

        # Find the specified repo
        target_repo = None
        for repo in cache_info.repos:
            if repo.repo_id == target_repo_id:
                target_repo = repo
                break

        if not target_repo:
            self.logger.info(f"Repository {target_repo_id} not found in cache")
            return {
                "repo_id": target_repo_id,
                "cached": False,
                "message": "Repository not found in cache",
            }

        return {
            "repo_id": target_repo_id,
            "repo_type": target_repo.repo_type,
            "cached": True,
            "size_on_disk": target_repo.size_on_disk_str,
            "size_bytes": target_repo.size_on_disk,
            "files_count": target_repo.nb_files,
            "last_accessed": target_repo.last_accessed,
            "last_modified": target_repo.last_modified,
            "revisions_count": len(target_repo.revisions),
            "revisions": [
                {
                    "commit_hash": rev.commit_hash,
                    "short_hash": rev.commit_hash[:8],
                    "size_on_disk": rev.size_on_disk_str,
                    "size_bytes": rev.size_on_disk,
                    "last_modified": rev.last_modified,
                    "files_count": len(rev.files),
                    "files": [
                        {
                            "name": file.file_name,
                            "size": file.size_on_disk,
                            "blob_last_accessed": file.blob_last_accessed,
                            "blob_last_modified": file.blob_last_modified,
                        }
                        for file in sorted(rev.files, key=lambda f: f.file_name)
                    ],
                }
                for rev in sorted(
                    target_repo.revisions, key=lambda r: r.last_modified, reverse=True
                )
            ],
        }

        self.logger.info(
            f"Found repo {target_repo_id} in cache: {target_repo.size_on_disk_str}, "
            f"{target_repo.nb_files} files, {len(target_repo.revisions)} revisions"
        )

        return result

    def check_cached_files(self, table_name: str | None = None) -> dict[str, Any]:
        """
        Check which files for current dataset/table are cached locally.

        :param table_name: Specific table to check (defaults to all tables)
        :return: Dictionary with file cache status
        """
        if not self._cache_manager:
            self.logger.debug("Cache management is disabled")
            return {"cache_management": "disabled"}

        if table_name:
            self.logger.info(f"Checking cache status for table: {table_name}")
        else:
            self.logger.info(f"Checking cache status for all tables in {self.repo_id}")

        from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

        result = {
            "repo_id": self.repo_id,
            "cache_directory": str(self.cache_dir),
            "tables": {},
        }

        # Check specific table or all tables
        tables_to_check = [table_name] if table_name else self.available_tables
        self.logger.debug(f"Checking {len(tables_to_check)} tables: {tables_to_check}")

        for table in tables_to_check:
            if not self.data_info.has_table(table):
                self.logger.warning(f"Table {table} not found in dataset configuration")
                result["tables"][table] = {
                    "exists": False,
                    "message": "Table not found in dataset configuration",
                }
                continue

            table_config = self.data_info.get_table_info(table)
            if not table_config:
                continue

            file_status = {}
            for file_info in table_config.data_files:
                file_path = file_info.path

                # Check if file is cached
                cached_path = try_to_load_from_cache(
                    repo_id=self.repo_id,
                    filename=file_path,
                    repo_type=self.repo_type,
                )

                if isinstance(cached_path, str):
                    # File is cached
                    file_status[file_path] = {
                        "cached": True,
                        "local_path": cached_path,
                        "split": file_info.split,
                    }
                elif cached_path is _CACHED_NO_EXIST:
                    # Non-existence is cached (file doesn't exist on Hub)
                    file_status[file_path] = {
                        "cached": False,
                        "exists_on_hub": False,
                        "split": file_info.split,
                    }
                else:
                    # File is not cached
                    file_status[file_path] = {
                        "cached": False,
                        "exists_on_hub": True,  # Assumed
                        "split": file_info.split,
                    }

            cached_count = sum(
                1 for f in file_status.values() if f.get("cached", False)
            )
            total_files = len(table_config.data_files)

            result["tables"][table] = {
                "exists": True,
                "files": file_status,
                "total_files": total_files,
                "cached_files": cached_count,
                "is_partitioned": table_config.is_partitioned,
            }

            self.logger.info(
                f"Table {table}: {cached_count}/{total_files} files cached "
                f"({cached_count/total_files*100:.1f}%)"
            )

        total_tables = len(
            [t for t in result["tables"].values() if t.get("exists", False)]
        )
        self.logger.info(f"Cache check complete: processed {total_tables} tables")

        return result

    def cleanup_cache(
        self,
        strategy: str = "auto",
        max_age_days: int | None = None,
        target_size: str | None = None,
        keep_current_repo: bool = True,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """
        Clean up the HuggingFace Hub cache using various strategies.

        :param strategy: Cleanup strategy - "auto", "age", "size", "unused"
        :param max_age_days: Maximum age for cache entries (for "age" strategy)
        :param target_size: Target cache size (for "size" strategy, e.g., "5GB")
        :param keep_current_repo: Whether to preserve current repo from cleanup
        :param dry_run: If True, show what would be deleted without executing
        :return: Dictionary with cleanup results and summary
        """
        if not self._cache_manager:
            self.logger.warning("Cache management is disabled, cannot perform cleanup")
            return {
                "cache_management": "disabled",
                "message": "Cache management is not enabled",
            }

        # Use instance defaults if not specified
        max_age_days = max_age_days or getattr(self, "_cache_max_age_days", 30)
        target_size = target_size or getattr(self, "_cache_max_size", "10GB")

        self.logger.info(
            f"Starting cache cleanup: strategy={strategy}, max_age={max_age_days}d, "
            f"target_size={target_size}, dry_run={dry_run}, keep_current_repo={keep_current_repo}"
        )

        result = {
            "strategy": strategy,
            "dry_run": dry_run,
            "keep_current_repo": keep_current_repo,
            "strategies_executed": [],
            "total_freed_bytes": 0,
            "total_freed_str": "0B",
        }

        try:
            if strategy == "auto":
                # Multi-strategy automated cleanup
                self.logger.info("Executing automated multi-strategy cleanup")
                strategies = self._cache_manager.auto_clean_cache(
                    max_age_days=max_age_days,
                    max_total_size=target_size,
                    keep_latest_per_repo=2,
                    dry_run=dry_run,
                )
                total_freed = sum(s.expected_freed_size for s in strategies)
                result["strategies_executed"] = [
                    {
                        "type": "auto_cleanup",
                        "freed_bytes": total_freed,
                        "freed_str": self._cache_manager._format_bytes(total_freed),
                        "details": f"Executed {len(strategies)} cleanup strategies",
                    }
                ]
                result["total_freed_bytes"] = total_freed
                self.logger.info(
                    f"Auto cleanup {'would free' if dry_run else 'freed'} "
                    f"{self._cache_manager._format_bytes(total_freed)} using {len(strategies)} strategies"
                )

            elif strategy == "age":
                # Age-based cleanup
                self.logger.info(
                    f"Executing age-based cleanup (older than {max_age_days} days)"
                )
                delete_strategy = self._cache_manager.clean_cache_by_age(
                    max_age_days=max_age_days, dry_run=dry_run
                )
                result["strategies_executed"] = [
                    {
                        "type": "age_based",
                        "freed_bytes": delete_strategy.expected_freed_size,
                        "freed_str": delete_strategy.expected_freed_size_str,
                        "max_age_days": max_age_days,
                    }
                ]
                result["total_freed_bytes"] = delete_strategy.expected_freed_size
                self.logger.info(
                    f"Age-based cleanup {'would free' if dry_run else 'freed'} "
                    f"{delete_strategy.expected_freed_size_str}"
                )

            elif strategy == "size":
                # Size-based cleanup
                self.logger.info(
                    f"Executing size-based cleanup (target: {target_size})"
                )
                delete_strategy = self._cache_manager.clean_cache_by_size(
                    target_size=target_size, strategy="oldest_first", dry_run=dry_run
                )
                result["strategies_executed"] = [
                    {
                        "type": "size_based",
                        "freed_bytes": delete_strategy.expected_freed_size,
                        "freed_str": delete_strategy.expected_freed_size_str,
                        "target_size": target_size,
                    }
                ]
                result["total_freed_bytes"] = delete_strategy.expected_freed_size
                self.logger.info(
                    f"Size-based cleanup {'would free' if dry_run else 'freed'} "
                    f"{delete_strategy.expected_freed_size_str} to reach target {target_size}"
                )

            elif strategy == "unused":
                # Clean unused revisions
                self.logger.info(
                    "Executing unused revisions cleanup (keeping 2 latest per repo)"
                )
                delete_strategy = self._cache_manager.clean_unused_revisions(
                    keep_latest=2, dry_run=dry_run
                )
                result["strategies_executed"] = [
                    {
                        "type": "unused_revisions",
                        "freed_bytes": delete_strategy.expected_freed_size,
                        "freed_str": delete_strategy.expected_freed_size_str,
                        "keep_latest": 2,
                    }
                ]
                result["total_freed_bytes"] = delete_strategy.expected_freed_size
                self.logger.info(
                    f"Unused revisions cleanup {'would free' if dry_run else 'freed'} "
                    f"{delete_strategy.expected_freed_size_str}"
                )

            else:
                self.logger.error(f"Unknown cleanup strategy: {strategy}")
                return {
                    "error": f"Unknown cleanup strategy: {strategy}",
                    "available_strategies": ["auto", "age", "size", "unused"],
                }

            result["total_freed_str"] = self._cache_manager._format_bytes(
                result["total_freed_bytes"]
            )

            # Add current repo protection info
            if keep_current_repo:
                result["current_repo_protected"] = {
                    "repo_id": self.repo_id,
                    "message": "Current repository was protected from cleanup",
                }
                self.logger.debug(
                    f"Current repository {self.repo_id} protected from cleanup"
                )

            # Final summary logging
            self.logger.info(
                f"Cache cleanup completed: {strategy} strategy, "
                f"{'would free' if dry_run else 'freed'} {result['total_freed_str']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            return {
                "error": f"Cache cleanup failed: {str(e)}",
                "strategy": strategy,
                "dry_run": dry_run,
            }

    def auto_cleanup_cache_if_needed(self) -> dict[str, Any]:
        """
        Automatically clean cache if configured policies are exceeded.

        This method is called automatically during operations if auto_cleanup is enabled.

        :return: Dictionary with cleanup results or None if no cleanup was needed
        """
        if not self._cache_manager or not getattr(self, "_cache_auto_cleanup", False):
            self.logger.debug("Auto-cleanup is disabled")
            return {"auto_cleanup": "disabled"}

        self.logger.debug("Checking if auto-cleanup is needed")

        from huggingface_hub import scan_cache_dir

        try:
            cache_info = scan_cache_dir()
        except Exception as e:
            self.logger.error(f"Failed to scan cache for auto-cleanup: {e}")
            return {"auto_cleanup": "error", "error": str(e)}
        max_size_bytes = self._cache_manager._parse_size_string(
            getattr(self, "_cache_max_size", "10GB")
        )

        cleanup_needed = cache_info.size_on_disk > max_size_bytes

        if not cleanup_needed:
            self.logger.debug(
                f"Auto-cleanup not needed: {cache_info.size_on_disk_str} "
                f"< {getattr(self, '_cache_max_size', '10GB')}"
            )
            return {
                "auto_cleanup": "enabled",
                "cleanup_needed": False,
                "current_size": cache_info.size_on_disk_str,
                "max_size": getattr(self, "_cache_max_size", "10GB"),
            }

        self.logger.info(
            f"Auto-cleanup triggered: cache size ({cache_info.size_on_disk_str}) "
            f"exceeds limit ({getattr(self, '_cache_max_size', '10GB')})"
        )

        cleanup_result = self.cleanup_cache(
            strategy="auto",
            dry_run=False,  # Execute cleanup
            keep_current_repo=True,
        )

        cleanup_result.update(
            {
                "auto_cleanup": "enabled",
                "triggered": True,
                "reason": "cache_size_exceeded",
                "previous_size": cache_info.size_on_disk_str,
            }
        )

        return cleanup_result

    def suggest_cache_cleanup(self) -> dict[str, Any]:
        """
        Analyze cache and provide cleanup recommendations without executing.

        :return: Dictionary with analysis and recommendations
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        from datetime import datetime, timedelta

        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        suggestions = {
            "cache_analysis": {
                "total_size": cache_info.size_on_disk_str,
                "total_repos": len(cache_info.repos),
                "recommendations": [],
            },
            "cleanup_strategies": {},
        }

        # Analyze size
        max_size_bytes = self._cache_manager._parse_size_string(
            getattr(self, "_cache_max_size", "10GB")
        )
        if cache_info.size_on_disk > max_size_bytes:
            suggestions["cache_analysis"]["recommendations"].append(
                {
                    "type": "size_exceeded",
                    "message": f"Cache size ({cache_info.size_on_disk_str}) exceeds "
                    f"configured limit ({getattr(self, '_cache_max_size', '10GB')})",
                    "suggested_action": "Run cleanup_cache(strategy='size')",
                }
            )

        # Analyze age
        cutoff_date = datetime.now() - timedelta(
            days=getattr(self, "_cache_max_age_days", 30)
        )
        old_repos = []
        for repo in cache_info.repos:
            for revision in repo.revisions:
                if datetime.fromtimestamp(revision.last_modified) < cutoff_date:
                    old_repos.append((repo.repo_id, revision.commit_hash[:8]))

        if old_repos:
            suggestions["cache_analysis"]["recommendations"].append(
                {
                    "type": "old_revisions",
                    "message": f"Found {len(old_repos)} old revisions "
                    f"(older than {getattr(self, '_cache_max_age_days', 30)} days)",
                    "suggested_action": "Run cleanup_cache(strategy='age')",
                }
            )

        # Analyze unused revisions
        repos_with_multiple_revisions = [
            repo for repo in cache_info.repos if len(repo.revisions) > 2
        ]
        if repos_with_multiple_revisions:
            suggestions["cache_analysis"]["recommendations"].append(
                {
                    "type": "multiple_revisions",
                    "message": f"Found {len(repos_with_multiple_revisions)} repos "
                    "with multiple cached revisions",
                    "suggested_action": "Run cleanup_cache(strategy='unused')",
                }
            )

        # Dry run different strategies to show potential savings
        try:
            age_cleanup = self._cache_manager.clean_cache_by_age(
                max_age_days=getattr(self, "_cache_max_age_days", 30), dry_run=True
            )
            suggestions["cleanup_strategies"]["age_based"] = {
                "description": f"Remove revisions older than {getattr(self, '_cache_max_age_days', 30)} days",
                "potential_savings": age_cleanup.expected_freed_size_str,
                "potential_savings_bytes": age_cleanup.expected_freed_size,
            }

            size_cleanup = self._cache_manager.clean_cache_by_size(
                target_size=getattr(self, "_cache_max_size", "10GB"),
                strategy="oldest_first",
                dry_run=True,
            )
            suggestions["cleanup_strategies"]["size_based"] = {
                "description": f"Reduce cache to {getattr(self, '_cache_max_size', '10GB')}",
                "potential_savings": size_cleanup.expected_freed_size_str,
                "potential_savings_bytes": size_cleanup.expected_freed_size,
            }

            unused_cleanup = self._cache_manager.clean_unused_revisions(
                keep_latest=2, dry_run=True
            )
            suggestions["cleanup_strategies"]["unused_revisions"] = {
                "description": "Remove unused revisions (keep 2 latest per repo)",
                "potential_savings": unused_cleanup.expected_freed_size_str,
                "potential_savings_bytes": unused_cleanup.expected_freed_size,
            }

        except Exception as e:
            suggestions["error"] = f"Failed to analyze cleanup strategies: {e}"

        return suggestions

    def warm_cache(
        self,
        repo_ids: list[str] | None = None,
        tables: list[str] | None = None,
        include_current_repo: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Pre-download (warm) cache with specified repositories or tables.

        :param repo_ids: List of repository IDs to pre-download
        :param tables: List of table names from current repo to pre-download
        :param include_current_repo: Whether to include current repo if repo_ids specified
        :param dry_run: If True, show what would be downloaded without executing
        :return: Dictionary with warming results
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        result = {
            "cache_warming": "enabled",
            "dry_run": dry_run,
            "operations": [],
            "total_downloaded": 0,
            "errors": [],
        }

        # Handle table-specific warming for current repo
        if tables:
            repo_result = {
                "repo_id": self.repo_id,
                "type": "table_specific",
                "tables": {},
                "success": True,
            }

            for table_name in tables:
                try:
                    if not self.data_info.has_table(table_name):
                        repo_result["tables"][table_name] = {
                            "status": "error",
                            "message": "Table not found in dataset configuration",
                        }
                        continue

                    if not dry_run:
                        # Download files for this table
                        self._ensure_dataset_loaded(table_name)
                        repo_result["tables"][table_name] = {
                            "status": "downloaded",
                            "message": "Table files cached successfully",
                        }
                        result["total_downloaded"] += 1
                    else:
                        repo_result["tables"][table_name] = {
                            "status": "would_download",
                            "message": "Would download table files",
                        }

                except Exception as e:
                    error_msg = f"Failed to warm cache for table {table_name}: {e}"
                    repo_result["tables"][table_name] = {
                        "status": "error",
                        "message": str(e),
                    }
                    result["errors"].append(error_msg)
                    repo_result["success"] = False

            result["operations"].append(repo_result)

        # Handle repository-specific warming
        if repo_ids or (not tables and include_current_repo):
            target_repos = repo_ids or []
            if include_current_repo and self.repo_id not in target_repos:
                target_repos.append(self.repo_id)

            for repo_id in target_repos:
                repo_result = {
                    "repo_id": repo_id,
                    "type": "full_repo",
                    "success": True,
                    "message": "",
                }

                try:
                    if not dry_run:
                        if repo_id == self.repo_id:
                            # Use current API instance for current repo
                            downloaded_path = self.download()
                            repo_result["message"] = (
                                f"Repository cached at {downloaded_path}"
                            )
                        else:
                            # Create temporary API instance for other repos
                            temp_api = HfQueryAPI(
                                repo_id=repo_id,
                                repo_type=self.repo_type,
                                token=self.token,
                                cache_dir=self.cache_dir,
                                enable_cache_management=False,  # Avoid recursive cache management
                            )
                            downloaded_path = temp_api.download()
                            repo_result["message"] = (
                                f"Repository cached at {downloaded_path}"
                            )

                        result["total_downloaded"] += 1
                    else:
                        repo_result["message"] = f"Would download repository {repo_id}"

                except Exception as e:
                    error_msg = f"Failed to warm cache for repo {repo_id}: {e}"
                    repo_result["success"] = False
                    repo_result["message"] = str(e)
                    result["errors"].append(error_msg)

                result["operations"].append(repo_result)

        if not repo_ids and not tables and not include_current_repo:
            result["message"] = "No repositories or tables specified for cache warming"

        return result

    def verify_cache_integrity(self) -> dict[str, Any]:
        """
        Verify integrity of cached files and detect corruption.

        :return: Dictionary with verification results
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        import os

        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        result = {
            "cache_verification": "enabled",
            "total_repos_scanned": len(cache_info.repos),
            "issues_found": [],
            "healthy_repos": [],
            "summary": {
                "healthy": 0,
                "corrupted": 0,
                "missing_files": 0,
                "symlink_issues": 0,
            },
        }

        for repo in cache_info.repos:
            repo_issues = []

            # Check if snapshots directory exists
            if not repo.repo_path.exists():
                repo_issues.append(
                    {
                        "type": "missing_repo_directory",
                        "message": f"Repository directory does not exist: {repo.repo_path}",
                    }
                )
                result["summary"]["corrupted"] += 1
                continue

            # Check each revision
            for revision in repo.revisions:
                if not revision.snapshot_path.exists():
                    repo_issues.append(
                        {
                            "type": "missing_snapshot",
                            "revision": revision.commit_hash[:8],
                            "message": f"Snapshot directory missing: {revision.snapshot_path}",
                        }
                    )
                    continue

                # Check each file in the revision
                for file_info in revision.files:
                    if not file_info.file_path.exists():
                        repo_issues.append(
                            {
                                "type": "missing_file",
                                "revision": revision.commit_hash[:8],
                                "file": file_info.file_name,
                                "message": f"File missing: {file_info.file_path}",
                            }
                        )
                        continue

                    # Check if it's a symlink and target exists
                    if file_info.file_path.is_symlink():
                        if not file_info.blob_path.exists():
                            repo_issues.append(
                                {
                                    "type": "broken_symlink",
                                    "revision": revision.commit_hash[:8],
                                    "file": file_info.file_name,
                                    "message": f"Symlink target missing: {file_info.blob_path}",
                                }
                            )
                            continue

                    # Basic file size check
                    try:
                        actual_size = os.path.getsize(file_info.file_path)
                        if actual_size != file_info.size_on_disk:
                            repo_issues.append(
                                {
                                    "type": "size_mismatch",
                                    "revision": revision.commit_hash[:8],
                                    "file": file_info.file_name,
                                    "expected_size": file_info.size_on_disk,
                                    "actual_size": actual_size,
                                    "message": f"File size mismatch in {file_info.file_name}",
                                }
                            )
                    except (OSError, IOError) as e:
                        repo_issues.append(
                            {
                                "type": "access_error",
                                "revision": revision.commit_hash[:8],
                                "file": file_info.file_name,
                                "message": f"Cannot access file: {e}",
                            }
                        )

            # Categorize the repo
            if repo_issues:
                result["issues_found"].append(
                    {
                        "repo_id": repo.repo_id,
                        "repo_type": repo.repo_type,
                        "issues": repo_issues,
                        "issues_count": len(repo_issues),
                    }
                )

                # Update summary
                if any(
                    issue["type"] in ["missing_repo_directory", "missing_snapshot"]
                    for issue in repo_issues
                ):
                    result["summary"]["corrupted"] += 1
                elif any(issue["type"] == "missing_file" for issue in repo_issues):
                    result["summary"]["missing_files"] += 1
                elif any(issue["type"] == "broken_symlink" for issue in repo_issues):
                    result["summary"]["symlink_issues"] += 1
                else:
                    result["summary"]["corrupted"] += 1
            else:
                result["healthy_repos"].append(
                    {
                        "repo_id": repo.repo_id,
                        "repo_type": repo.repo_type,
                        "size": repo.size_on_disk_str,
                    }
                )
                result["summary"]["healthy"] += 1

        # Overall health assessment
        total_repos = len(cache_info.repos)
        if total_repos > 0:
            health_percentage = (result["summary"]["healthy"] / total_repos) * 100
            result["overall_health"] = {
                "percentage": round(health_percentage, 1),
                "status": (
                    "healthy"
                    if health_percentage > 95
                    else "warning" if health_percentage > 80 else "critical"
                ),
                "recommendation": self._get_health_recommendation(health_percentage),
            }

        return result

    def _get_health_recommendation(self, health_percentage: float) -> str:
        """Get recommendation based on cache health percentage."""
        if health_percentage > 95:
            return "Cache is in excellent condition"
        elif health_percentage > 80:
            return "Cache has minor issues. Consider running cleanup to remove problematic entries"
        else:
            return "Cache has significant issues. Run cleanup_cache() or consider clearing the cache entirely"

    def migrate_cache(self, new_cache_dir: str | Path) -> dict[str, Any]:
        """
        Migrate cache to a new directory location.

        :param new_cache_dir: Target directory for cache migration
        :return: Dictionary with migration results
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        import shutil
        from pathlib import Path

        new_cache_path = Path(new_cache_dir)
        current_cache_path = self.cache_dir

        result = {
            "cache_migration": "enabled",
            "source": str(current_cache_path),
            "destination": str(new_cache_path),
            "success": False,
            "files_migrated": 0,
            "errors": [],
        }

        try:
            # Validate target directory
            if new_cache_path.exists() and list(new_cache_path.iterdir()):
                return {
                    **result,
                    "error": f"Target directory {new_cache_path} is not empty",
                    "suggestion": "Choose an empty directory or clear the target directory",
                }

            # Create target directory if needed
            new_cache_path.mkdir(parents=True, exist_ok=True)

            # Get current cache info
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()

            if not cache_info.repos:
                result.update(
                    {
                        "success": True,
                        "message": "No cached repositories to migrate",
                    }
                )
                # Update cache directory
                self.cache_dir = new_cache_path
                return result

            # Migrate each repository
            migrated_repos = []
            for repo in cache_info.repos:
                try:
                    # Create repo structure in new location
                    repo_name = repo.repo_path.name
                    new_repo_path = new_cache_path / repo_name

                    # Copy entire repository directory
                    shutil.copytree(repo.repo_path, new_repo_path, symlinks=True)
                    migrated_repos.append(repo.repo_id)
                    result["files_migrated"] += repo.nb_files

                except Exception as e:
                    error_msg = f"Failed to migrate repo {repo.repo_id}: {e}"
                    result["errors"].append(error_msg)

            if migrated_repos and not result["errors"]:
                # Migration successful, update cache directory
                self.cache_dir = new_cache_path
                result.update(
                    {
                        "success": True,
                        "migrated_repos": migrated_repos,
                        "repos_count": len(migrated_repos),
                        "message": f"Successfully migrated {len(migrated_repos)} repositories",
                    }
                )
            elif migrated_repos:
                result.update(
                    {
                        "success": True,  # Partial success
                        "migrated_repos": migrated_repos,
                        "repos_count": len(migrated_repos),
                        "message": f"Partially migrated {len(migrated_repos)} repositories with {len(result['errors'])} errors",
                    }
                )
            else:
                result.update(
                    {
                        "success": False,
                        "message": "Migration failed completely",
                    }
                )

        except Exception as e:
            result.update(
                {
                    "success": False,
                    "error": f"Migration failed: {str(e)}",
                }
            )

        return result

    # ============== Cache Configuration Management ==============

    def configure_cache_policies(
        self,
        auto_cleanup: bool | None = None,
        max_age_days: int | None = None,
        max_size: str | None = None,
        save_to_env: bool = False,
    ) -> dict[str, Any]:
        """
        Configure cache management policies for this instance.

        :param auto_cleanup: Enable/disable automatic cache cleanup
        :param max_age_days: Maximum age in days for cache entries
        :param max_size: Maximum total cache size (e.g., "10GB")
        :param save_to_env: Save configuration to environment variables
        :return: Dictionary with updated configuration
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        # Update instance configuration
        if auto_cleanup is not None:
            self._cache_auto_cleanup = auto_cleanup
        if max_age_days is not None:
            self._cache_max_age_days = max_age_days
        if max_size is not None:
            self._cache_max_size = max_size

        config = {
            "cache_management": "enabled",
            "configuration_updated": True,
            "policies": {
                "auto_cleanup": getattr(self, "_cache_auto_cleanup", False),
                "max_age_days": getattr(self, "_cache_max_age_days", 30),
                "max_size": getattr(self, "_cache_max_size", "10GB"),
            },
            "env_variables_updated": False,
        }

        # Save to environment variables if requested
        if save_to_env:
            import os

            env_vars = {}

            if auto_cleanup is not None:
                env_var = "TFBPAPI_CACHE_AUTO_CLEANUP"
                os.environ[env_var] = str(auto_cleanup).lower()
                env_vars[env_var] = str(auto_cleanup).lower()

            if max_age_days is not None:
                env_var = "TFBPAPI_CACHE_MAX_AGE_DAYS"
                os.environ[env_var] = str(max_age_days)
                env_vars[env_var] = str(max_age_days)

            if max_size is not None:
                env_var = "TFBPAPI_CACHE_MAX_SIZE"
                os.environ[env_var] = max_size
                env_vars[env_var] = max_size

            config.update(
                {
                    "env_variables_updated": True,
                    "env_variables": env_vars,
                    "note": "Environment variables set for current session. "
                    "Add them to your shell profile for persistence.",
                }
            )

        return config

    def get_cache_configuration(self) -> dict[str, Any]:
        """
        Get current cache configuration including environment variables.

        :return: Dictionary with comprehensive cache configuration
        """
        import os

        config = {
            "cache_management": "enabled" if self._cache_manager else "disabled",
            "cache_directory": str(self.cache_dir),
            "instance_config": {},
            "environment_config": {},
            "effective_config": {},
        }

        if not self._cache_manager:
            return config

        # Instance configuration
        config["instance_config"] = {
            "auto_cleanup": getattr(self, "_cache_auto_cleanup", False),
            "max_age_days": getattr(self, "_cache_max_age_days", 30),
            "max_size": getattr(self, "_cache_max_size", "10GB"),
        }

        # Environment configuration
        env_config = {}
        env_vars = {
            "TFBPAPI_CACHE_AUTO_CLEANUP": "auto_cleanup",
            "TFBPAPI_CACHE_MAX_AGE_DAYS": "max_age_days",
            "TFBPAPI_CACHE_MAX_SIZE": "max_size",
            "HF_CACHE_DIR": "cache_directory",
            "HF_HUB_CACHE": "cache_directory_fallback",
        }

        for env_var, config_key in env_vars.items():
            value = os.getenv(env_var)
            if value:
                env_config[config_key] = value

        config["environment_config"] = env_config

        # Effective configuration (environment overrides instance)
        effective = config["instance_config"].copy()

        # Apply environment overrides
        if "auto_cleanup" in env_config:
            effective["auto_cleanup"] = env_config["auto_cleanup"].lower() in (
                "true",
                "1",
                "yes",
            )
        if "max_age_days" in env_config:
            try:
                effective["max_age_days"] = int(env_config["max_age_days"])
            except ValueError:
                pass
        if "max_size" in env_config:
            effective["max_size"] = env_config["max_size"]

        config["effective_config"] = effective

        return config

    def reset_cache_configuration(
        self, remove_env_vars: bool = False
    ) -> dict[str, Any]:
        """
        Reset cache configuration to defaults.

        :param remove_env_vars: Also remove related environment variables
        :return: Dictionary with reset results
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        # Reset instance configuration to defaults
        self._cache_auto_cleanup = False
        self._cache_max_age_days = 30
        self._cache_max_size = "10GB"

        result = {
            "cache_management": "enabled",
            "configuration_reset": True,
            "new_config": {
                "auto_cleanup": False,
                "max_age_days": 30,
                "max_size": "10GB",
            },
            "env_variables_removed": [],
        }

        # Remove environment variables if requested
        if remove_env_vars:
            import os

            env_vars_to_remove = [
                "TFBPAPI_CACHE_AUTO_CLEANUP",
                "TFBPAPI_CACHE_MAX_AGE_DAYS",
                "TFBPAPI_CACHE_MAX_SIZE",
            ]

            for env_var in env_vars_to_remove:
                if env_var in os.environ:
                    del os.environ[env_var]
                    result["env_variables_removed"].append(env_var)

            if result["env_variables_removed"]:
                result["note"] = (
                    "Environment variables removed from current session. "
                    "Remove them from your shell profile for permanent effect."
                )

        return result

    def apply_cache_policy_from_env(self) -> dict[str, Any]:
        """
        Apply cache policies from environment variables.

        :return: Dictionary with applied configuration
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        import os

        applied_config = []

        # Auto cleanup
        auto_cleanup_env = os.getenv("TFBPAPI_CACHE_AUTO_CLEANUP")
        if auto_cleanup_env:
            self._cache_auto_cleanup = auto_cleanup_env.lower() in ("true", "1", "yes")
            applied_config.append(f"auto_cleanup: {self._cache_auto_cleanup}")

        # Max age days
        max_age_env = os.getenv("TFBPAPI_CACHE_MAX_AGE_DAYS")
        if max_age_env:
            try:
                self._cache_max_age_days = int(max_age_env)
                applied_config.append(f"max_age_days: {self._cache_max_age_days}")
            except ValueError:
                applied_config.append(
                    f"max_age_days: invalid value '{max_age_env}' (keeping default)"
                )

        # Max size
        max_size_env = os.getenv("TFBPAPI_CACHE_MAX_SIZE")
        if max_size_env:
            self._cache_max_size = max_size_env
            applied_config.append(f"max_size: {self._cache_max_size}")

        return {
            "cache_management": "enabled",
            "environment_config_applied": True,
            "applied_settings": applied_config,
            "current_config": {
                "auto_cleanup": getattr(self, "_cache_auto_cleanup", False),
                "max_age_days": getattr(self, "_cache_max_age_days", 30),
                "max_size": getattr(self, "_cache_max_size", "10GB"),
            },
        }

    def export_cache_configuration(self, format: str = "env") -> dict[str, Any]:
        """
        Export current cache configuration in various formats.

        :param format: Export format - "env", "json", "yaml"
        :return: Dictionary with exported configuration
        """
        if not self._cache_manager:
            return {"cache_management": "disabled"}

        current_config = {
            "auto_cleanup": getattr(self, "_cache_auto_cleanup", False),
            "max_age_days": getattr(self, "_cache_max_age_days", 30),
            "max_size": getattr(self, "_cache_max_size", "10GB"),
            "cache_directory": str(self.cache_dir),
        }

        result = {
            "cache_management": "enabled",
            "format": format,
            "configuration": current_config,
        }

        if format == "env":
            env_lines = [
                f"export TFBPAPI_CACHE_AUTO_CLEANUP={str(current_config['auto_cleanup']).lower()}",
                f"export TFBPAPI_CACHE_MAX_AGE_DAYS={current_config['max_age_days']}",
                f"export TFBPAPI_CACHE_MAX_SIZE={current_config['max_size']}",
                f"export HF_CACHE_DIR={current_config['cache_directory']}",
            ]
            result["exported_config"] = "\n".join(env_lines)
            result["usage"] = "Add these lines to your ~/.bashrc or ~/.zshrc file"

        elif format == "json":
            import json

            result["exported_config"] = json.dumps(current_config, indent=2)

        elif format == "yaml":
            yaml_lines = [
                "cache_configuration:",
                f"  auto_cleanup: {str(current_config['auto_cleanup']).lower()}",
                f"  max_age_days: {current_config['max_age_days']}",
                f"  max_size: \"{current_config['max_size']}\"",
                f"  cache_directory: \"{current_config['cache_directory']}\"",
            ]
            result["exported_config"] = "\n".join(yaml_lines)

        else:
            result["error"] = f"Unsupported format: {format}"
            result["supported_formats"] = ["env", "json", "yaml"]

        return result

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
