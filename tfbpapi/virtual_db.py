"""
VirtualDB provides a SQL query interface across heterogeneous datasets.

A developer creates huggingface repos with datacards. Datacard specifications
specific to tfbpapi can be found at
https://brentlab.github.io/tfbpapi/huggingface_datacard/. Next, a developer can create
a virtualDB configuration file that describes which huggingface repos and datasets to
use, a set of common fields, datasets that contain comparative analytics, and more.
VirtualDB, this code, then uses DuckDB to construct tables and views are
which are lazily created over Parquet files which are cached locally. VirtualDB uses
the information in the datacard to create metadata views which describe sample level
features. Derived columns are attached to both the metadata and full data views. Any
comparative analysis datasets are also parsed and joined to the primary datasets'
metadata views. The expectation is that a developer will use this interface to write
SQL queries against the views to provide an API to downstream users and applications.

Example Usage::

    from tfbpapi.virtual_db import VirtualDB

    vdb = VirtualDB("config.yaml", token=token)

    # Discover views
    vdb.tables()
    vdb.describe("harbison")

    # Raw SQL
    df = vdb.query("SELECT * FROM harbison WHERE sample_id = 42")

    # Parameterized SQL
    df = vdb.query(
        "SELECT * FROM harbison_meta WHERE carbon_source = $cs",
        cs="glucose",
    )

    # Prepared queries
    vdb.prepare("sig", "SELECT * FROM harbison_meta LIMIT $n")
    df = vdb.query("sig", n=10)

"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from tfbpapi.datacard import DataCard
from tfbpapi.models import MetadataConfig

logger = logging.getLogger(__name__)


def get_nested_value(data: dict | list, path: str) -> Any:
    """
    Navigate nested dict/list using dot notation.

    Handles missing intermediate keys gracefully by returning None.
    When an intermediate value is a list of dicts, extracts the
    remaining path from each item and returns a list of results.

    :param data: Dictionary or list of dicts to navigate
    :param path: Dot-separated path (e.g., "media.carbon_source.compound")
    :return: Value at path, list of values, or None if not found

    :raises TypeError: If an unexpected type is encountered during navigation of the
        dict/list structure according to the provided path.

    Example -- dict input::

        >>> get_nested_value({"media": {"name": "YPD"}}, "media.name")
        'YPD'

    Example -- list-of-dicts at an intermediate node::

        >>> data = {
        ...     "media": {
        ...         "carbon_source": [
        ...             {"compound": "glucose"},
        ...         ]
        ...     }
        ... }
        >>> get_nested_value(data, "media.carbon_source.compound")
        ['glucose']

    """
    if not isinstance(data, (dict, list)):
        return None

    # If top-level data is a list, extract path from each item
    if isinstance(data, list):
        results = []
        for item in data:
            if isinstance(item, dict):
                val = get_nested_value(item, path)
                if val is not None:
                    results.append(val)
        return results if results else None

    keys = path.split(".")
    current = data

    for i, key in enumerate(keys):
        if isinstance(current, dict):
            if key not in current:
                logger.warning(
                    "Key '%s' not found at path '%s' (current keys: %s)",
                    key,
                    ".".join(keys[: i + 1]),
                    list(current.keys()),
                )
                return None
            current = current[key]
        elif isinstance(current, list):
            # Extract the remaining path from each list item
            remaining_path = ".".join(keys[i:])
            results = []
            for item in current:
                if isinstance(item, dict):
                    val = get_nested_value(item, remaining_path)
                    if val is not None:
                        results.append(val)
            return results if results else None
        else:
            error_msg = (
                f"Unexpected type '{type(current).__name__}' at "
                f"path '{'.'.join(keys[:i])}'; expected dict or "
                f"list of dicts"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

    return current


@lru_cache(maxsize=32)
def _cached_datacard(repo_id: str, token: str | None = None) -> Any:
    """
    Return a cached DataCard instance.

    :param repo_id: HuggingFace repository ID
    :param token: Optional HuggingFace token
    :return: DataCard instance

    """
    return DataCard(repo_id, token=token)


class VirtualDB:
    """
    A query interface across heterogeneous datasets.

    DuckDB views are lazily registered over Parquet files on first
    ``query()`` call. The user writes SQL against named views.

    :ivar config: Validated MetadataConfig
    :ivar token: Optional HuggingFace token

    """

    def __init__(
        self,
        config_path: Path | str,
        token: str | None = None,
        duckdb_connection: duckdb.DuckDBPyConnection | None = None,
        views_registered: bool = False,
        lazy: bool = True,
    ):
        """
        Initialize VirtualDB with configuration.

        :param config_path: Path to YAML configuration file
        :param token: Optional HuggingFace token for private datasets
        :param duckdb_connection: Optional DuckDB connection. If provided, views will be
            registered on this connection instead of creating a new in-memory database.
            Note that this provides a method of using a persistent database file. If
            this isn't provided, then the duckDB connection is in-memory.
        :param views_registered: If True, skip view registration (assumes views are
            already registered on the provided duckdb_connection). This is useful when
            reusing a connection across multiple VirtualDB instances with the same
            config.
        :param lazy: If True, delay DuckDB connection and view registration until first
            query. Set to False to register views immediately on initialization. This is
            intended to be used when creating a persistent duckDB connection. If the
            views are registered immediately on initialization, then for any other
            instances of VirtualDB that are initialized with the same duckDB connection
            and config, the views will already be registered and available for querying.
        :raises FileNotFoundError: If config file does not exist
        :raises ValueError: If configuration is invalid or if views_registered=True is
            set when lazy=False

        """
        if not lazy and views_registered:
            raise ValueError(
                "Cannot set views_registered=True when lazy=False. "
                "If lazy=False, views will be registered immediately on initialization."
            )
        self.config = MetadataConfig.from_yaml(config_path)
        self.token = token

        # Instantiate without creating a connection, if no connection is provided.
        # the connection is created when needed by calling self._ensure_sql_views()
        self._conn: duckdb.DuckDBPyConnection | None = duckdb_connection
        self._views_registered = views_registered

        # db_name -> (repo_id, config_name)
        self._db_name_map = self._build_db_name_map()

        # Prepared queries: name -> sql
        self._prepared_queries: dict[str, str] = {}

        # If not lazy, create the DuckDB connection and register views immediately.
        if not lazy:
            self._ensure_sql_views()

    @property
    def _db(self) -> duckdb.DuckDBPyConnection:
        """Return the DuckDB connection, asserting it is initialized."""
        assert self._conn is not None, (
            "DuckDB connection not initialized. " "Call _ensure_sql_views() first."
        )
        return self._conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, sql: str, **params: Any) -> pd.DataFrame:
        """
        Execute SQL or a prepared query and return a DataFrame.

        If *sql* matches a registered prepared-query name the stored
        SQL template is used instead. Keyword arguments are passed as
        named parameters to DuckDB.

        :param sql: Raw SQL string **or** name of a prepared query
        :param params: Named parameters (DuckDB ``$name`` syntax)
        :return: Query result as a pandas DataFrame

        Examples::

            # Raw SQL
            df = vdb.query("SELECT * FROM harbison LIMIT 5")

            # With parameters
            df = vdb.query(
                "SELECT * FROM harbison_meta WHERE carbon_source = $cs",
                cs="glucose",
            )

            # Prepared query
            vdb.prepare("top", "SELECT * FROM harbison_meta LIMIT $n")
            df = vdb.query("top", n=10)

        """
        self._ensure_sql_views()

        # param `sql` may be a prepared query name, a raw sql statement, or
        # a parameterized sql statement that is not prepared. If it exists as a key
        # in the _prepared_queries dict, we use the prepared sql. Otherwise, we
        # use the sql as passed to query().
        resolved = self._prepared_queries.get(sql, sql)
        if params:
            return self._db.execute(resolved, params).fetchdf()
        return self._db.execute(resolved).fetchdf()

    def prepare(self, name: str, sql: str, overwrite: bool = False) -> None:
        """
        Register a named parameterized query for later use.

                Parameters use DuckDB ``$name`` syntax.

                :param name: Query name (must not collide with a view name)
                :param sql: SQL template with ``$name`` parameters
                :param overwrite: If True, overwrite existing prepared query
                    with same name
                :raises ValueError: If *name* collides with an existing view
        con
                Example::

                    vdb.prepare("glucose_regs", '''
                        SELECT regulator_symbol, COUNT(*) AS n
                        FROM harbison_meta
                        WHERE carbon_source = $cs
                        GROUP BY regulator_symbol
                        HAVING n >= $min_n
                    ''')
                    df = vdb.query("glucose_regs", cs="glucose", min_n=2)

        """
        self._ensure_sql_views()
        if name in self._list_views() and not overwrite:
            error_msg = (
                f"Prepared-query name '{name}' collides with "
                f"an existing view. Choose a different name or set "
                f"overwrite=True."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._prepared_queries[name] = sql

    def tables(self) -> list[str]:
        """
        Return sorted list of registered view names.

        :return: Sorted list of view names

        """
        self._ensure_sql_views()
        return sorted(self._list_views())

    def describe(self, table: str | None = None) -> pd.DataFrame:
        """
        Describe column names and types for one or all views.

        :param table: View name, or None for all views
        :return: DataFrame with columns ``table``, ``column_name``,
                 ``column_type``

        """
        self._ensure_sql_views()
        if table is not None:
            df = self._db.execute(f"DESCRIBE {table}").fetchdf()
            df.insert(0, "table", table)
            return df

        frames = []
        for view in sorted(self._list_views()):
            df = self._db.execute(f"DESCRIBE {view}").fetchdf()
            df.insert(0, "table", view)
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["table", "column_name", "column_type"])
        return pd.concat(frames, ignore_index=True)

    def get_fields(self, table: str | None = None) -> list[str]:
        """
        Return column names for a view or all unique columns.

        :param table: View name, or None for all views
        :return: Sorted list of column names

        """
        self._ensure_sql_views()
        if table is not None:
            cols = self._db.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{table}'"
            ).fetchdf()
            return sorted(cols["column_name"].tolist())

        all_cols: set[str] = set()
        for view in self._list_views():
            cols = self._db.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{view}'"
            ).fetchdf()
            all_cols.update(cols["column_name"].tolist())
        return sorted(all_cols)

    def get_common_fields(self) -> list[str]:
        """
        Return columns present in ALL primary ``_meta`` views.

        Primary dataset views are those without ``links`` in their
        config (i.e. not comparative datasets).

        :return: Sorted list of common column names

        """
        self._ensure_sql_views()
        meta_views = self._get_primary_meta_view_names()
        if not meta_views:
            return []

        sets = []
        for view in meta_views:
            cols = self._db.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{view}'"
            ).fetchdf()
            sets.append(set(cols["column_name"].tolist()))

        common = set.intersection(*sets)
        return sorted(common)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_sql_views(self) -> None:
        """Create DuckDB connection and register all views on first call."""
        if self._views_registered:
            return
        self._conn = duckdb.connect(":memory:")
        self._register_all_views()
        self._views_registered = True

    def _register_all_views(self) -> None:
        """Orchestrate view registration in dependency order."""
        # 1. Raw per-dataset views (internal __<db_name>_parquet
        # plus public <db_name> for primary datasets only)
        for db_name, (repo_id, config_name) in self._db_name_map.items():
            comparative = self._is_comparative(repo_id, config_name)
            self._register_raw_view(
                db_name,
                repo_id,
                config_name,
                parquet_only=comparative,
            )

        # 2. Metadata views for primary datasets (<db_name>_meta)
        # This is based on the metadata defined in the datacard,
        # and includes any additional derived columns based on the
        # virtualDB config passed in at initialization. Note that
        # this is joined onto the raw view in the next step.
        for db_name, (repo_id, config_name) in self._db_name_map.items():
            if not self._is_comparative(repo_id, config_name):
                self._register_meta_view(db_name, repo_id, config_name)

        # 3. Replace primary raw views with join to _meta so
        # derived columns (e.g. carbon_source) are available
        for db_name, (repo_id, config_name) in self._db_name_map.items():
            if not self._is_comparative(repo_id, config_name):
                self._enrich_raw_view(db_name)

        # 4. Comparative expanded views (pre-parsed composite IDs)
        # These build directly on __<db_name>_parquet since
        # comparative datasets have no _meta or enriched raw view.
        for db_name, (repo_id, config_name) in self._db_name_map.items():
            repo_cfg = self.config.repositories.get(repo_id)
            if not repo_cfg or not repo_cfg.dataset:
                continue
            ds_cfg = repo_cfg.dataset.get(config_name)
            if ds_cfg and ds_cfg.links:
                self._register_comparative_expanded_view(db_name, ds_cfg)

    # ------------------------------------------------------------------
    # db_name mapping
    # ------------------------------------------------------------------

    def _build_db_name_map(self) -> dict[str, tuple[str, str]]:
        """
        Build mapping from resolved db_name to (repo_id, config_name).

        :return: Dict mapping db_name -> (repo_id, config_name)

        """
        mapping: dict[str, tuple[str, str]] = {}
        for repo_id, repo_cfg in self.config.repositories.items():
            if not repo_cfg.dataset:
                continue
            for config_name, ds_cfg in repo_cfg.dataset.items():
                resolved = ds_cfg.db_name or config_name
                mapping[resolved] = (repo_id, config_name)
        return mapping

    # ------------------------------------------------------------------
    # Parquet file resolution
    # ------------------------------------------------------------------

    def _resolve_parquet_files(self, repo_id: str, config_name: str) -> list[str]:
        """
        Download (or locate cached) Parquet files for a dataset config.

        Uses ``huggingface_hub.snapshot_download`` with the file patterns
        from the DataCard.

        :param repo_id: HuggingFace repository ID
        :param config_name: Dataset configuration name
        :return: List of absolute paths to Parquet files

        """
        card = DataCard(repo_id, token=self.token)
        config = card.get_config(config_name)
        if not config:
            logger.warning(
                "Config '%s' not found in repo '%s'",
                config_name,
                repo_id,
            )
            return []

        file_patterns = [df.path for df in config.data_files]

        from huggingface_hub import snapshot_download

        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=file_patterns,
            token=self.token,
        )

        parquet_files: list[str] = []
        for pattern in file_patterns:
            file_path = Path(downloaded_path) / pattern
            if file_path.exists() and file_path.suffix == ".parquet":
                parquet_files.append(str(file_path))
            elif "*" in pattern:
                base = Path(downloaded_path)
                parquet_files.extend(
                    str(f) for f in base.glob(pattern) if f.suffix == ".parquet"
                )
            else:
                parent_dir = Path(downloaded_path) / Path(pattern).parent
                if parent_dir.exists():
                    parquet_files.extend(str(f) for f in parent_dir.glob("*.parquet"))

        return parquet_files

    # ------------------------------------------------------------------
    # View registration helpers
    # ------------------------------------------------------------------

    def _register_raw_view(
        self,
        db_name: str,
        repo_id: str,
        config_name: str,
        *,
        parquet_only: bool = False,
    ) -> None:
        """
        Register a raw DuckDB view over Parquet files.

        Creates an internal ``__<db_name>_parquet`` view that reads
        directly from the Parquet files. For primary datasets, also
        creates a public ``<db_name>`` view (initially identical)
        that may later be replaced by ``_enrich_raw_view``.

        For comparative datasets, only the internal parquet view is
        created; the public view is the ``_expanded`` view instead.

        :param db_name: View name
        :param repo_id: Repository ID
        :param config_name: Configuration name
        :param parquet_only: If True, only create the internal
            ``__<db_name>_parquet`` view (no public ``<db_name>``).

        """
        files = self._resolve_parquet_files(repo_id, config_name)
        if not files:
            logger.warning(
                "No parquet files for %s/%s -- skipping view '%s'",
                repo_id,
                config_name,
                db_name,
            )
            return

        files_sql = ", ".join(f"'{f}'" for f in files)
        parquet_sql = f"SELECT * FROM read_parquet([{files_sql}])"
        self._db.execute(
            f"CREATE OR REPLACE VIEW __{db_name}_parquet AS " f"{parquet_sql}"
        )
        if not parquet_only:
            self._db.execute(
                f"CREATE OR REPLACE VIEW {db_name} AS "
                f"SELECT * FROM __{db_name}_parquet"
            )

    def _register_meta_view(self, db_name: str, repo_id: str, config_name: str) -> None:
        """
        Register a ``<db_name>_meta`` view with one row per sample_id.

        Includes raw metadata columns from the DataCard plus any derived columns from
        config property mappings (resolved against DataCard definitions with factor
        aliases applied).

        :param db_name: Base view name for the primary dataset
        :param repo_id: Repository ID
        :param config_name: Configuration name

        """
        parquet_view = f"__{db_name}_parquet"
        if not self._view_exists(parquet_view):
            return

        meta_cols = self._resolve_metadata_fields(repo_id, config_name)
        prop_result = self._resolve_property_columns(repo_id, config_name)

        if prop_result is not None:
            derived_exprs, prop_raw_cols = prop_result
            # Raw cols = metadata_fields + any source fields needed
            # by property mappings
            if meta_cols is not None:
                raw = list(dict.fromkeys(["sample_id"] + meta_cols + prop_raw_cols))
            else:
                raw = list(dict.fromkeys(["sample_id"] + prop_raw_cols))

            raw_sql = ", ".join(raw)

            # Outer SELECT: raw cols + derived expressions
            outer_parts = list(raw) + derived_exprs
            outer_sql = ", ".join(outer_parts)

            self._db.execute(
                f"CREATE OR REPLACE VIEW {db_name}_meta AS "
                f"SELECT DISTINCT {outer_sql} "
                f"FROM ("
                f"SELECT DISTINCT {raw_sql} "
                f"FROM {parquet_view}"
                f") AS __raw"
            )
        elif meta_cols is not None:
            # Fallback: metadata_fields only, no property mappings
            cols = list(dict.fromkeys(["sample_id"] + meta_cols))
            cols_sql = ", ".join(cols)
            self._db.execute(
                f"CREATE OR REPLACE VIEW {db_name}_meta AS "
                f"SELECT DISTINCT {cols_sql} "
                f"FROM {parquet_view}"
            )
        else:
            # No metadata_fields at all -- all columns are metadata
            self._db.execute(
                f"CREATE OR REPLACE VIEW {db_name}_meta AS "
                f"SELECT DISTINCT * FROM {parquet_view}"
            )

    def _enrich_raw_view(self, db_name: str) -> None:
        """
        Replace a primary raw view with a join to its ``_meta`` view.

        If ``<db_name>_meta`` has derived columns not present in the
        raw parquet view, recreates ``<db_name>`` as a join so derived
        columns (e.g. ``carbon_source``) appear alongside measurement
        data.

        :param db_name: Base view name for the primary dataset

        """
        meta_name = f"{db_name}_meta"
        parquet_name = f"__{db_name}_parquet"
        if not self._view_exists(meta_name) or not self._view_exists(parquet_name):
            return

        raw_cols = set(self._get_view_columns(parquet_name))
        meta_cols = set(self._get_view_columns(meta_name))
        extra_cols = meta_cols - raw_cols

        if not extra_cols:
            return

        extra_select = ", ".join(f"m.{c}" for c in sorted(extra_cols))
        self._db.execute(
            f"CREATE OR REPLACE VIEW {db_name} AS "
            f"SELECT r.*, {extra_select} "
            f"FROM {parquet_name} r "
            f"JOIN {meta_name} m USING (sample_id)"
        )

    def _get_view_columns(self, view: str) -> list[str]:
        """Return column names for a view."""
        df = self._db.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{view}'"
        ).fetchdf()
        return df["column_name"].tolist()

    def _resolve_metadata_fields(
        self, repo_id: str, config_name: str
    ) -> list[str] | None:
        """
        Get the metadata_fields list from the DataCard config.

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :return: List of metadata field names, or None if not specified

        """
        try:
            card = _cached_datacard(repo_id, token=self.token)
            config = card.get_config(config_name)
            if config and config.metadata_fields:
                return list(config.metadata_fields)
        except Exception:
            logger.debug(
                "Could not resolve metadata_fields for %s/%s",
                repo_id,
                config_name,
            )
        return None

    def _resolve_alias(self, col: str, value: str) -> str:
        """
        Apply factor alias to a value if one is configured.

        :param col: Column name (e.g., "carbon_source")
        :param value: Raw value (e.g., "D-glucose")
        :return: Canonical alias (e.g., "glucose") or original value

        """
        aliases = self.config.factor_aliases.get(col)
        if not aliases:
            return value
        lower_val = str(value).lower()
        for canonical, actuals in aliases.items():
            if lower_val in [str(a).lower() for a in actuals]:
                return canonical
        return value

    def _resolve_property_columns(
        self,
        repo_id: str,
        config_name: str,
    ) -> tuple[list[str], list[str]] | None:
        """
        Build SQL column expressions for derived property columns.

        Resolves config property mappings against the DataCard to
        produce SQL expressions that add derived columns to the
        ``_meta`` view.

        :param repo_id: Repository ID
        :param config_name: Configuration name
        :return: Tuple of (sql_expressions, raw_cols_needed) or None
            if no property mappings are configured.
            ``sql_expressions`` are SQL fragments like
            ``"'glucose' AS carbon_source"`` or
            ``"CASE WHEN ... END AS carbon_source"``.
            ``raw_cols_needed`` are raw parquet column names that must
            be present in the inner SELECT.

        """
        mappings = self.config.get_property_mappings(repo_id, config_name)
        if not mappings:
            return None

        expressions: list[str] = []
        raw_cols: set[str] = set()

        try:
            card = _cached_datacard(repo_id, token=self.token)
        except Exception as exc:
            logger.warning(
                "Could not load DataCard for %s: %s",
                repo_id,
                exc,
            )
            return None

        for key, mapping in mappings.items():
            if mapping.expression is not None:
                # Type D: expression
                expressions.append(f"({mapping.expression}) AS {key}")
                continue

            if mapping.field is not None and mapping.path is None:
                # Type A: field-only (alias or no-op)
                raw_cols.add(mapping.field)
                if key == mapping.field:
                    # no-op -- column already present as raw col
                    pass
                else:
                    expressions.append(f"{mapping.field} AS {key}")
                continue

            if mapping.field is not None and mapping.path is not None:
                # Type B: field + path -- resolve from definitions
                raw_cols.add(mapping.field)
                expr = self._build_field_path_expr(
                    key,
                    mapping.field,
                    mapping.path,
                    mapping.dtype,
                    config_name,
                    card,
                )
                if expr is not None:
                    expressions.append(expr)
                continue

            if mapping.field is None and mapping.path is not None:
                # Type C: path-only -- constant from config
                expr = self._build_path_only_expr(
                    key,
                    mapping.path,
                    mapping.dtype,
                    config_name,
                    card,
                )
                if expr is not None:
                    expressions.append(expr)
                continue

        if not expressions and not raw_cols:
            return None

        return expressions, sorted(raw_cols)

    def _build_field_path_expr(
        self,
        key: str,
        field: str,
        path: str,
        dtype: str | None,
        config_name: str,
        card: Any,
    ) -> str | None:
        """
        Build a SQL expression for a field+path property mapping.

        Resolves each definition value via ``get_nested_value``,
        applies factor aliases, and returns either a constant or
        a CASE WHEN expression.

        :param key: Output column name
        :param field: Source field in parquet (e.g., "condition")
        :param path: Dot-notation path within definitions
        :param dtype: Optional data type ("numeric", "string", "bool")
        :param config_name: Configuration name
        :param card: DataCard instance
        :return: SQL expression string, or None on failure

        """
        try:
            defs = card.get_field_definitions(config_name, field)
        except Exception as exc:
            logger.warning(
                "Could not get definitions for field '%s' " "in config '%s': %s",
                field,
                config_name,
                exc,
            )
            return None

        if not defs:
            return None

        # Resolve each definition value
        value_map: dict[str, str] = {}
        for def_key, definition in defs.items():
            raw = get_nested_value(definition, path)
            if raw is None:
                logger.debug(
                    "Path '%s' resolved to None for " "definition key '%s' (keys: %s)",
                    path,
                    def_key,
                    (
                        list(definition.keys())
                        if isinstance(definition, dict)
                        else type(definition).__name__
                    ),
                )
                continue
            # Handle list results (e.g., carbon_source returns
            # [{"compound": "D-glucose"}])
            if isinstance(raw, list):
                raw = raw[0] if len(raw) == 1 else ", ".join(str(v) for v in raw)
            resolved = self._resolve_alias(key, str(raw))
            value_map[str(def_key)] = resolved

        if not value_map:
            return None

        # If all values are the same, emit a constant
        unique_vals = set(value_map.values())
        if len(unique_vals) == 1:
            val = next(iter(unique_vals))
            return self._literal_expr(key, val, dtype)

        # Otherwise, build CASE WHEN
        whens = []
        for def_key, resolved in value_map.items():
            escaped_key = def_key.replace("'", "''")
            escaped_val = resolved.replace("'", "''")
            whens.append(f"WHEN {field} = '{escaped_key}' " f"THEN '{escaped_val}'")
        case_sql = " ".join(whens)
        missing = self.config.missing_value_labels.get(key)
        if missing is not None:
            escaped_missing = missing.replace("'", "''")
            expr = f"CASE {case_sql} " f"ELSE '{escaped_missing}' END"
        else:
            expr = f"CASE {case_sql} ELSE NULL END"
        if dtype == "numeric":
            expr = f"CAST({expr} AS DOUBLE)"
        return f"{expr} AS {key}"

    def _build_path_only_expr(
        self,
        key: str,
        path: str,
        dtype: str | None,
        config_name: str,
        card: Any,
    ) -> str | None:
        """
        Build a constant column expression for a path-only mapping.

        Resolves a single value from the DataCard's raw model_extra,
        which preserves the full dict structure (including any
        ``experimental_conditions`` wrapper).

        :param key: Output column name
        :param path: Dot-notation path (may include
            ``experimental_conditions.`` prefix)
        :param dtype: Optional data type
        :param config_name: Configuration name
        :param card: DataCard instance
        :return: SQL literal expression, or None on failure

        """
        # Build merged dict from top-level + config-level model_extra.
        # This preserves keys like "experimental_conditions" that
        # get_experimental_conditions() would strip.
        merged: dict[str, Any] = {}
        try:
            top_extra = card.dataset_card.model_extra
            if isinstance(top_extra, dict):
                merged.update(top_extra)
            config_obj = card.get_config(config_name)
            if config_obj and isinstance(config_obj.model_extra, dict):
                merged.update(config_obj.model_extra)
        except Exception:
            logger.debug(
                "Could not get model_extra for %s/%s",
                card.repo_id if hasattr(card, "repo_id") else "?",
                config_name,
            )
            return None

        if not merged:
            return None

        raw = get_nested_value(merged, path)
        if raw is None:
            logger.debug(
                "Path '%s' resolved to None in model_extra for "
                "%s/%s. Available keys: %s",
                path,
                card.repo_id if hasattr(card, "repo_id") else "?",
                config_name,
                list(merged.keys()),
            )
            return None

        if isinstance(raw, list):
            raw = raw[0] if len(raw) == 1 else ", ".join(str(v) for v in raw)

        resolved = self._resolve_alias(key, str(raw))
        return self._literal_expr(key, resolved, dtype)

    @staticmethod
    def _literal_expr(key: str, value: str, dtype: str | None) -> str:
        """
        Build a SQL literal expression with optional type cast.

        :param key: Column alias
        :param value: Literal value
        :param dtype: Optional type ("numeric", "string", "bool")
        :return: SQL expression

        """
        escaped = value.replace("'", "''")
        if dtype == "numeric":
            return f"CAST('{escaped}' AS DOUBLE) AS {key}"
        return f"'{escaped}' AS {key}"

    def _register_comparative_expanded_view(
        self,
        db_name: str,
        ds_cfg: Any,
    ) -> None:
        """
        Create ``<db_name>_expanded`` view with parsed composite ID cols.

        For each link_field in the dataset config, adds two columns:

        - ``<link_field>_source`` -- the ``repo_id;config_name`` prefix,
          aliased to the configured ``db_name`` when available.
        - ``<link_field>_id`` -- the sample_id component.

        :param db_name: Base view name for the comparative dataset
        :param ds_cfg: DatasetVirtualDBConfig with ``links``

        """
        parquet_view = f"__{db_name}_parquet"
        if not self._view_exists(parquet_view):
            return

        extra_cols = []
        for link_field, primaries in ds_cfg.links.items():
            # _id column: third component of composite ID
            id_col = f"{link_field}_id"
            extra_cols.append(f"SPLIT_PART({link_field}, ';', 3) " f"AS {id_col}")

            # _source column: first two components, aliased
            # to db_name when the pair is in the config
            raw_expr = (
                f"SPLIT_PART({link_field}, ';', 1) || ';' "
                f"|| SPLIT_PART({link_field}, ';', 2)"
            )
            whens = []
            for pair in primaries:
                repo_id, config_name = pair[0], pair[1]
                alias = self._get_db_name_for(repo_id, config_name)
                if alias:
                    key = f"{repo_id};{config_name}".replace("'", "''")
                    whens.append(f"WHEN '{key}' THEN '{alias}'")
            if whens:
                case_sql = " ".join(whens)
                source_expr = f"CASE {raw_expr} {case_sql} " f"ELSE {raw_expr} END"
            else:
                source_expr = raw_expr
            source_col = f"{link_field}_source"
            extra_cols.append(f"{source_expr} AS {source_col}")

        if not extra_cols:
            return

        cols_sql = ", ".join(extra_cols)
        self._db.execute(
            f"CREATE OR REPLACE VIEW {db_name}_expanded AS "
            f"SELECT *, {cols_sql} FROM {parquet_view}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_comparative(self, repo_id: str, config_name: str) -> bool:
        """Return True if the dataset has links (i.e. is comparative)."""
        repo_cfg = self.config.repositories.get(repo_id)
        if not repo_cfg or not repo_cfg.dataset:
            return False
        ds_cfg = repo_cfg.dataset.get(config_name)
        return bool(ds_cfg and ds_cfg.links)

    def _list_views(self) -> list[str]:
        """Return list of public views (excludes internal __ prefixed)."""
        df = self._db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'VIEW'"
        ).fetchdf()
        return [n for n in df["table_name"].tolist() if not n.startswith("__")]

    def _view_exists(self, name: str) -> bool:
        """Check whether a view is registered (including internal)."""
        df = self._db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'VIEW' "
            f"AND table_name = '{name}'"
        ).fetchdf()
        return len(df) > 0

    def _get_primary_view_names(self) -> list[str]:
        """
        Return db_names of primary (non-comparative) raw views.

        A primary dataset is one whose config has no ``links``.

        """
        names = []
        for db_name, (repo_id, config_name) in self._db_name_map.items():
            if not self._is_comparative(repo_id, config_name):
                if self._view_exists(db_name):
                    names.append(db_name)
        return sorted(names)

    def _get_primary_meta_view_names(self) -> list[str]:
        """Return names of primary ``_meta`` views."""
        return [
            f"{n}_meta"
            for n in self._get_primary_view_names()
            if self._view_exists(f"{n}_meta")
        ]

    def _get_db_name_for(self, repo_id: str, config_name: str) -> str | None:
        """Resolve db_name for a (repo_id, config_name) pair."""
        for db_name, (r, c) in self._db_name_map.items():
            if r == repo_id and c == config_name:
                return db_name
        return None

    def __repr__(self) -> str:
        """String representation."""
        n_repos = len(self.config.repositories)
        n_datasets = len(self._db_name_map)
        if self._views_registered:
            n_views = len(self._list_views())
            return (
                f"VirtualDB({n_repos} repos, "
                f"{n_datasets} datasets, "
                f"{n_views} views)"
            )
        return (
            f"VirtualDB({n_repos} repos, "
            f"{n_datasets} datasets, views not yet registered)"
        )
