import logging
import re
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from .constants import CACHE_DIR, SQL_FILTER_KEYWORDS
from .errors import InvalidFilterFieldError
from .HfCacheManager import HfCacheManager


class HfQueryAPI(HfCacheManager):
    """Minimal Hugging Face API client focused on metadata retrieval."""

    def __init__(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
        token: str | None = None,
        cache_dir: str | Path | None = None,
        duckdb_conn: duckdb.DuckDBPyConnection = duckdb.connect(":memory:"),
    ):
        """
        Initialize the minimal HF Query API client.

        :param repo_id: Repository identifier (e.g., "user/dataset")
        :param repo_type: Type of repository ("dataset", "model", "space")
        :param token: HuggingFace token for authentication
        :param cache_dir: HF cache_dir for downloads

        """
        self._duckdb_conn = duckdb_conn

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
        # dict structure:
        #   {config_name: "SQL WHERE clause", ...}
        self._table_filters: dict[str, str] = {}

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

    def _get_explicit_metadata(self, config, table_name: str) -> pd.DataFrame:
        """Helper function to handle explicit metadata configurations."""
        sql = f"SELECT * FROM {table_name}"
        return self.duckdb_conn.execute(sql).fetchdf()

    def _get_embedded_metadata(self, config, table_name: str) -> pd.DataFrame:
        """Helper function to handle embedded metadata configurations."""
        if config.metadata_fields is None:
            raise ValueError(f"Config {config.config_name} has no metadata fields")
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
        return self.duckdb_conn.execute(sql).fetchdf()

    def _validate_metadata_fields(
        self, config_name: str, field_names: list[str]
    ) -> None:
        """
        Validate that field names exist in the config's columns or joinable metadata.

        Checks both:
        1. The config's own columns
        2. Columns from metadata configs that have join_keys defined

        :param config_name: Configuration name to validate against
        :param field_names: List of field names to validate
        :raises InvalidFilterFieldError: If any fields don't exist in available columns

        """
        if not field_names:
            return

        try:
            # Get columns from the base config
            base_columns = self._get_columns_from_config(config_name)
            available_fields = set(base_columns)

            # Add columns from any metadata configs with join_keys
            relationships = self.get_metadata_relationships()
            data_relationships = [
                r for r in relationships if r.data_config == config_name
            ]

            for rel in data_relationships:
                if rel.relationship_type == "explicit" and rel.join_keys:
                    # This metadata can be auto-joined, include its columns
                    metadata_columns = self._get_columns_from_config(
                        rel.metadata_config
                    )
                    available_fields.update(metadata_columns)

            # Check for invalid fields
            invalid_fields = [
                field for field in field_names if field not in available_fields
            ]

            if invalid_fields:
                raise InvalidFilterFieldError(
                    config_name=config_name,
                    invalid_fields=invalid_fields,
                    available_fields=list(available_fields),
                )
        except Exception as e:
            if isinstance(e, InvalidFilterFieldError):
                raise
            # If metadata retrieval fails for other reasons, log warning but allow
            self.logger.warning(
                f"Could not validate filter fields for {config_name}: {e}"
            )

    def _extract_fields_from_sql(self, sql_where: str) -> list[str]:
        """
        Extract potential field names from SQL WHERE clause.

        Uses a more robust approach to identify column references while avoiding string
        literals used as values.

        :param sql_where: SQL WHERE clause (without 'WHERE' keyword)
        :return: List of potential field names found in the SQL

        """
        if not sql_where.strip():
            return []

        field_names = set()

        # Tokenize the SQL to better understand context
        # This regex splits on key tokens while preserving them
        tokens = re.findall(
            r"""
            \bIN\s*\([^)]+\)|                    # IN clauses with content
            \bBETWEEN\s+\S+\s+AND\s+\S+|        # BETWEEN clauses
            (?:'[^']*')|(?:"[^"]*")|             # Quoted strings
            \b(?:AND|OR|NOT|IS|NULL|LIKE|BETWEEN|IN)\b|  # SQL keywords
            [=!<>]+|                             # Comparison operators
            [(),]|                               # Delimiters
            \b[a-zA-Z_][a-zA-Z0-9_]*\b|          # Identifiers
            \S+                                  # Other tokens
        """,
            sql_where,
            re.VERBOSE | re.IGNORECASE,
        )

        # Track the context to determine if an identifier is a field name or value
        i = 0
        while i < len(tokens):
            token = tokens[i].strip()
            if not token:
                i += 1
                continue

            # Skip IN clauses entirely - they contain values, not field names
            if re.match(r"\bIN\s*\(", token, re.IGNORECASE):
                i += 1
                continue

            # Skip BETWEEN clauses entirely - they contain values, not field names
            if re.match(r"\bBETWEEN\b", token, re.IGNORECASE):
                i += 1
                continue

            # Handle quoted strings - could be identifiers or
            # values depending on context
            if token.startswith(("'", '"')):
                # Extract the content inside quotes
                quoted_content = token[1:-1]

                # Find next significant token to determine context
                next_significant_token = None
                for j in range(i + 1, len(tokens)):
                    next_token = tokens[j].strip()
                    if next_token and next_token not in [" ", "\n", "\t"]:
                        next_significant_token = next_token
                        break

                # Check if this quoted string is a field name based on context
                is_quoted_field = False

                # Check what comes after this quoted string
                if next_significant_token:
                    # If followed by comparison operators or SQL keywords,
                    # it's a field name
                    if (
                        next_significant_token
                        in ["=", "!=", "<>", "<", ">", "<=", ">="]
                        or next_significant_token.upper() in ["IS", "LIKE", "NOT"]
                        or re.match(
                            r"\bBETWEEN\b", next_significant_token, re.IGNORECASE
                        )
                        or re.match(r"\bIN\s*\(", next_significant_token, re.IGNORECASE)
                    ):
                        is_quoted_field = True

                # Also check what comes before this quoted string
                if not is_quoted_field and i > 0:
                    # Find the previous significant token
                    prev_significant_token = None
                    for j in range(i - 1, -1, -1):
                        prev_token = tokens[j].strip()
                        if prev_token and prev_token not in [" ", "\n", "\t"]:
                            prev_significant_token = prev_token
                            break

                    # If preceded by a comparison operator, could be a field name
                    # But we need to be very careful not to treat string
                    # literals as field names
                    if prev_significant_token and prev_significant_token in [
                        "=",
                        "!=",
                        "<>",
                        "<",
                        ">",
                        "<=",
                        ">=",
                    ]:
                        # Only treat as field name if it looks like a
                        # database identifier
                        # AND doesn't look like a typical string value
                        if self._looks_like_identifier(
                            quoted_content
                        ) and self._looks_like_database_identifier(quoted_content):
                            is_quoted_field = True

                if is_quoted_field:
                    field_names.add(quoted_content)

                i += 1
                continue

            # Skip SQL keywords and operators
            if token.upper() in SQL_FILTER_KEYWORDS or token in [
                "=",
                "!=",
                "<>",
                "<",
                ">",
                "<=",
                ">=",
                "(",
                ")",
                ",",
            ]:
                i += 1
                continue

            # Skip numeric literals
            if re.match(r"^-?\d+(\.\d+)?$", token):
                i += 1
                continue

            # Check if this looks like an identifier (field name)
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", token):
                # Check the context - if the next non-whitespace token is a
                # comparison operator,
                # then this is likely a field name
                next_significant_token = None
                for j in range(i + 1, len(tokens)):
                    next_token = tokens[j].strip()
                    if next_token and next_token not in [" ", "\n", "\t"]:
                        next_significant_token = next_token
                        break

                # Check if followed by a comparison operator or SQL keyword that
                # indicates a field
                is_field = False

                if next_significant_token:
                    # Direct comparison operators
                    if next_significant_token in [
                        "=",
                        "!=",
                        "<>",
                        "<",
                        ">",
                        "<=",
                        ">=",
                    ]:
                        is_field = True
                    # SQL keywords that follow field names
                    elif next_significant_token.upper() in ["IS", "LIKE", "NOT"]:
                        is_field = True
                    # BETWEEN clause (could be just 'BETWEEN' or 'BETWEEN ... AND ...')
                    elif next_significant_token.upper() == "BETWEEN" or re.match(
                        r"\bBETWEEN\b", next_significant_token, re.IGNORECASE
                    ):
                        is_field = True
                    # IN clause (could be just 'IN' or 'IN (...)')
                    elif next_significant_token.upper() == "IN" or re.match(
                        r"\bIN\s*\(", next_significant_token, re.IGNORECASE
                    ):
                        is_field = True

                # If not a field yet, check other contexts
                if not is_field and i > 0:
                    # Find the previous significant token
                    prev_significant_token = None
                    for j in range(i - 1, -1, -1):
                        prev_token = tokens[j].strip()
                        if prev_token and prev_token not in [" ", "\n", "\t"]:
                            prev_significant_token = prev_token
                            break

                    # Case 1: After AND/OR and before an operator (original logic)
                    if (
                        prev_significant_token
                        and prev_significant_token.upper() in ["AND", "OR"]
                        and next_significant_token
                    ):
                        # Same checks as above
                        if next_significant_token in [
                            "=",
                            "!=",
                            "<>",
                            "<",
                            ">",
                            "<=",
                            ">=",
                        ]:
                            is_field = True
                        elif next_significant_token.upper() in ["IS", "LIKE", "NOT"]:
                            is_field = True
                        elif next_significant_token.upper() == "BETWEEN" or re.match(
                            r"\bBETWEEN\b", next_significant_token, re.IGNORECASE
                        ):
                            is_field = True
                        elif next_significant_token.upper() == "IN" or re.match(
                            r"\bIN\s*\(", next_significant_token, re.IGNORECASE
                        ):
                            is_field = True

                    # Case 2: After a comparison operator (second operand)
                    elif prev_significant_token and prev_significant_token in [
                        "=",
                        "!=",
                        "<>",
                        "<",
                        ">",
                        "<=",
                        ">=",
                    ]:
                        # But exclude function names (identifiers followed by '(')
                        if next_significant_token != "(":
                            is_field = True

                    # Case 3: After opening parenthesis (function parameter)
                    elif prev_significant_token == "(":
                        is_field = True

                if is_field:
                    field_names.add(token)

            i += 1

        return list(field_names)

    def _looks_like_identifier(self, content: str) -> bool:
        """
        Determine if quoted content looks like an identifier rather than a string
        literal.

        :param content: The content inside quotes
        :return: True if it looks like an identifier, False if it looks like a string
            literal

        """
        if not content:
            return False

        # Basic identifier pattern: starts with letter/underscore, contains only
        # alphanumeric/underscore
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", content):
            return True

        # Extended identifier pattern: could contain spaces if it's a column name
        # like "quoted field"
        # but not if it contains many special characters or looks like natural language
        if " " in content:
            # If it contains spaces, it should still look identifier-like
            # Allow simple cases like "quoted field" but not "this is a long string
            # value"
            words = content.split()
            if len(words) <= 3 and all(
                re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", word) for word in words
            ):
                return True
            return False

        return False

    def _looks_like_database_identifier(self, content: str) -> bool:
        """
        Determine if content looks like a database identifier (field/table name).

        This is more strict than _looks_like_identifier and helps distinguish between
        quoted identifiers like "field_name" and string values like "value1".

        :param content: The content to check
        :return: True if it looks like a database identifier

        """
        if not content:
            return False

        # Database identifiers typically:
        # 1. Don't start with numbers (field names rarely start with numbers)
        # 2. Often contain underscores or descriptive words
        # 3. Don't look like simple values

        # Reject if starts with a number (like "value1", "123abc")
        if content[0].isdigit():
            return False

        # Handle short simple values that could be literals or field names
        if len(content) <= 6 and re.match(r"^[a-z]+\d*$", content.lower()):
            # Allow common field name prefixes
            field_prefixes = ["field", "col", "column", "attr", "prop"]
            if any(content.lower().startswith(prefix) for prefix in field_prefixes):
                return True  # It's a valid field name like "field1", "col2"
            else:
                return False  # It's likely a simple value like "value", "test"

        # Accept if it contains underscore (common in field names)
        if "_" in content:
            return True

        # Accept if it has multiple words (like "quoted field")
        if " " in content:
            return True

        # Accept if it's a longer descriptive name
        if len(content) > 8:
            return True

        # Reject otherwise (likely a simple value)
        return False

    def get_metadata(
        self, config_name: str, refresh_cache: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve metadata as a DataFrame with actual metadata values for a specific
        config.

        Supports three types of metadata retrieval:
        1. Direct metadata configs: config_name is itself a metadata config
        2. Embedded metadata: config_name has metadata_fields defined
        3. Applied metadata: config_name appears in another metadata config's
        applies_to list

        For explicit metadata configs (types 1 & 3), returns all rows from metadata
        table.
        For embedded metadata (type 2), returns distinct combinations of metadata
        fields.

        :param config_name: Specific config name to retrieve metadata for
        :param refresh_cache: If True, force refresh from remote instead of using cache
        :return: DataFrame with metadata values for the specified config
        :raises ValueError: If config_name has no associated metadata
        :raises RuntimeError: If data loading fails for the config

        """
        # Get metadata relationships for this config
        relationships = self.get_metadata_relationships(refresh_cache=refresh_cache)

        relevant_relationships = None

        # First priority: data_config matches (config_name is a data config
        # with metadata)
        data_config_matches = [r for r in relationships if r.data_config == config_name]

        if data_config_matches:
            relevant_relationships = data_config_matches
        else:
            # Second priority: metadata_config matches (config_name is itself a
            # metadata config)
            metadata_config_matches = [
                r for r in relationships if r.metadata_config == config_name
            ]
            relevant_relationships = metadata_config_matches

        if not relevant_relationships:
            # Check what configs are available for helpful error message
            all_data_configs = {r.data_config for r in relationships}
            all_metadata_configs = {r.metadata_config for r in relationships}
            all_available = sorted(all_data_configs | all_metadata_configs)

            if not all_available:
                return pd.DataFrame()

            raise ValueError(
                f"Config '{config_name}' not found. "
                f"Available configs with metadata: {all_available}"
            )

        # Get the config object to process
        # For explicit relationships, use the metadata config
        # For embedded relationships, use the data config
        relationship = relevant_relationships[0]  # Use first relationship found

        if relationship.relationship_type == "explicit":
            # Find the metadata config
            if relationship.metadata_config == config_name:
                # config_name is itself a metadata config
                config = self.get_config(config_name)
            else:
                # config_name is a data config with metadata applied to it
                config = self.get_config(relationship.metadata_config)
        else:  # embedded
            # config_name is a data config with embedded metadata
            config = self.get_config(config_name)

        if not config:
            raise ValueError(f"Could not find config object for '{config_name}'")

        # Process the single configuration
        config_result = self._get_metadata_for_config(
            config, force_refresh=refresh_cache
        )

        if not config_result.get("success", False):
            raise RuntimeError(f"Failed to load data for config {config.config_name}")

        table_name = config_result.get("table_name")
        if not table_name:
            raise RuntimeError(f"No table name for config {config.config_name}")

        try:
            if relationship.relationship_type == "explicit":
                return self._get_explicit_metadata(config, table_name)
            else:  # embedded
                return self._get_embedded_metadata(config, table_name)
        except Exception as e:
            self.logger.error(f"Error querying metadata for {config.config_name}: {e}")
            raise

    def set_filter(self, config_name: str, **kwargs) -> None:
        """
        Set simple filters using keyword arguments.

        Converts keyword arguments to SQL WHERE clause and stores
        for automatic application. Validates that all filter fields
        exist in the config's metadata columns.

        :param config_name: Configuration name to apply filters to
        :param kwargs: Filter conditions as keyword arguments
            (e.g., time=15, mechanism="ZEV")
        :raises InvalidFilterFieldError: If any filter field doesn't exist
            in the metadata columns

        Example:
            api.set_filter("hackett_2020", time=15, mechanism="ZEV", restriction="P")
            # Equivalent to: WHERE time = 15 AND mechanism = 'ZEV' AND restriction = 'P'

        """
        if not kwargs:
            # If no kwargs provided, clear the filter
            self.clear_filter(config_name)
            return

        # Validate that all filter fields exist in metadata columns
        self._validate_metadata_fields(config_name, list(kwargs.keys()))

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

    def set_sql_filter(
        self, config_name: str, sql_where: str, validate_fields: bool = True
    ) -> None:
        """
        Set complex filters using SQL WHERE clause.

        Stores raw SQL WHERE clause for automatic application to queries.
        Validates that field references in the SQL exist in metadata columns
        unless validation is disabled.

        :param config_name: Configuration name to apply filters to
        :param sql_where: SQL WHERE clause (without the 'WHERE' keyword)
        :param validate_fields: Whether to validate field names (default: True)
        :raises InvalidFilterFieldError: If any field references don't exist
            in the metadata columns (when validate_fields=True)

        Example:
            api.set_sql_filter("hackett_2020", "time IN (15, 30) AND mechanism = 'ZEV'")
            # To skip validation for complex SQL:
            api.set_sql_filter("hackett_2020", "complex_expression(...)",
            validate_fields=False)

        """
        if not sql_where.strip():
            self.clear_filter(config_name)
            return

        # Validate fields if requested
        if validate_fields:
            extracted_fields = self._extract_fields_from_sql(sql_where)
            self._validate_metadata_fields(config_name, extracted_fields)

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

    def query(
        self,
        sql: str,
        config_name: str,
        refresh_cache: bool = False,
        auto_join_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Execute SQL query with automatic filter application and metadata joins.

        Loads the specified configuration, applies any stored filters,
        and executes the query. If auto_join_metadata is True (default),
        automatically detects metadata columns in the query and joins
        the appropriate metadata tables.

        :param sql: SQL query to execute
        :param config_name: Configuration name to query (table will be loaded if needed)
        :param refresh_cache: If True, force refresh from remote instead of using cache
        :param auto_join_metadata: If True, automatically join metadata tables
        when needed
        :return: DataFrame with query results
        :raises ValueError: If config_name not found or query fails

        Example:
            api.set_filter("hackett_2020", time=15, mechanism="ZEV")
            df = api.query("SELECT regulator_locus_tag, target_locus_tag
                FROM hackett_2020", "hackett_2020")
            # Automatically applies: WHERE time = 15 AND mechanism = 'ZEV'

        Example with metadata:
            # If cell_type is in experiment_metadata that applies_to binding_data:
            df = api.query("SELECT * FROM binding_data WHERE cell_type = 'K562'",
                          "binding_data")
            # Automatically joins experiment_metadata and filters by cell_type

        """
        # Validate config exists
        if config_name not in [c.config_name for c in self.configs]:
            available_configs = [c.config_name for c in self.configs]
            raise ValueError(
                f"Config '{config_name}' not found. "
                f"Available configs: {available_configs}"
            )

        # Load the configuration data
        config = self.get_config(config_name)
        if not config:
            raise ValueError(f"Could not retrieve config '{config_name}'")

        config_result = self._get_metadata_for_config(
            config, force_refresh=refresh_cache
        )
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

        # Handle automatic metadata joins if enabled
        if auto_join_metadata:
            # Extract column references from the query
            referenced_columns = self._extract_column_references(sql_with_table)

            # Also check for columns in stored filters
            if config_name in self._table_filters:
                filter_sql = self._table_filters[config_name]
                filter_columns = self._extract_column_references(filter_sql)
                referenced_columns.update(filter_columns)

            # Get columns from the base config
            base_columns = self._get_columns_from_config(config_name)

            # Find columns that aren't in the base config
            missing_columns = referenced_columns - base_columns

            if missing_columns:
                # Find metadata configs that might have these columns
                metadata_matches = self._find_metadata_for_columns(
                    config_name, missing_columns
                )

                if metadata_matches:
                    # Load metadata views and build JOIN clauses
                    metadata_joins = []
                    for metadata_config, join_keys in metadata_matches:
                        metadata_table = self._load_metadata_view(
                            metadata_config, refresh_cache=refresh_cache
                        )
                        metadata_joins.append(
                            (metadata_config, metadata_table, join_keys)
                        )
                        self.logger.info(
                            f"Auto-joining metadata '{metadata_config}' "
                            f"on keys: {join_keys}"
                        )

                    # Rewrite SQL to include JOINs
                    sql_with_table = self._build_join_sql(
                        sql_with_table, table_name, metadata_joins
                    )

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

    def _get_columns_from_config(self, config_name: str) -> set[str]:
        """
        Get all column names from a config's schema.

        :param config_name: Configuration name
        :return: Set of column names

        """
        config = self.get_config(config_name)
        if not config:
            return set()
        return {feature.name for feature in config.dataset_info.features}

    def _extract_column_references(self, sql: str) -> set[str]:
        """
        Extract column references from SQL query.

        Simple regex-based extraction that looks for identifiers in common SQL contexts.
        Not a full SQL parser, but good enough for most queries.

        :param sql: SQL query string
        :return: Set of potential column names

        """
        # Remove string literals to avoid false positives
        sql_no_strings = re.sub(r"'[^']*'", "", sql)
        sql_no_strings = re.sub(r'"[^"]*"', "", sql_no_strings)

        # Extract identifiers that appear in typical column contexts:
        # - After SELECT, WHERE, GROUP BY, ORDER BY, HAVING
        # - In comparisons (=, !=, <, >, etc.)
        # - After AS keyword
        column_patterns = [
            r"\b(?:SELECT|WHERE|AND|OR|ON|GROUP BY|ORDER BY|HAVING)\s+[\w.]+",
            r"[\w.]+\s*(?:=|!=|<>|<|>|<=|>=|LIKE|IN|IS)",
            r"AS\s+([\w.]+)",
        ]

        columns = set()
        for pattern in column_patterns:
            matches = re.finditer(pattern, sql_no_strings, re.IGNORECASE)
            for match in matches:
                # Extract the identifier part
                text = match.group(0)
                # Remove SQL keywords and operators
                for keyword in [
                    "SELECT",
                    "WHERE",
                    "AND",
                    "OR",
                    "ON",
                    "GROUP BY",
                    "ORDER BY",
                    "HAVING",
                    "AS",
                    "=",
                    "!=",
                    "<>",
                    "<",
                    ">",
                    "<=",
                    ">=",
                    "LIKE",
                    "IN",
                    "IS",
                ]:
                    text = re.sub(
                        r"\b" + keyword + r"\b", "", text, flags=re.IGNORECASE
                    )
                # Extract remaining identifiers
                identifiers = re.findall(r"\b[\w.]+\b", text)
                for ident in identifiers:
                    # Remove table prefixes (e.g., "table.column" -> "column")
                    if "." in ident:
                        columns.add(ident.split(".")[-1])
                    else:
                        columns.add(ident)

        # Filter out common SQL keywords and functions
        sql_keywords = {
            "SELECT",
            "FROM",
            "WHERE",
            "AND",
            "OR",
            "NOT",
            "IN",
            "IS",
            "NULL",
            "AS",
            "ON",
            "JOIN",
            "LEFT",
            "RIGHT",
            "INNER",
            "OUTER",
            "GROUP",
            "BY",
            "ORDER",
            "HAVING",
            "LIMIT",
            "OFFSET",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "CASE",
            "WHEN",
            "THEN",
            "ELSE",
            "END",
            "CAST",
            "TRUE",
            "FALSE",
        }
        columns = {c for c in columns if c.upper() not in sql_keywords}

        return columns

    def _find_metadata_for_columns(
        self, config_name: str, columns: set[str]
    ) -> list[tuple[str, list[str]]]:
        """
        Find metadata configs that contain the specified columns.

        :param config_name: Data config name being queried
        :param columns: Set of column names to search for
        :return: List of tuples (metadata_config_name, join_keys)

        """
        relationships = self.get_metadata_relationships()
        data_relationships = [r for r in relationships if r.data_config == config_name]

        metadata_matches = []
        for rel in data_relationships:
            if rel.relationship_type == "embedded":
                # Skip embedded metadata - columns are already in the data table
                continue

            # Get metadata config schema
            metadata_columns = self._get_columns_from_config(rel.metadata_config)

            # Check if any of the queried columns are in this metadata
            if columns & metadata_columns:
                if rel.join_keys:
                    metadata_matches.append((rel.metadata_config, rel.join_keys))
                else:
                    # Log warning if columns match but no join keys defined
                    self.logger.warning(
                        f"Columns {columns & metadata_columns} found in metadata "
                        f"config '{rel.metadata_config}' but no join_keys defined. "
                        f"Cannot automatically join. Please add join_keys to datacard."
                    )

        return metadata_matches

    def _load_metadata_view(
        self, metadata_config_name: str, refresh_cache: bool = False
    ) -> str:
        """
        Load metadata config into DuckDB and return the table name.

        :param metadata_config_name: Metadata config to load
        :param refresh_cache: Whether to refresh cache
        :return: Table name in DuckDB

        """
        config = self.get_config(metadata_config_name)
        if not config:
            raise ValueError(f"Metadata config '{metadata_config_name}' not found")

        config_result = self._get_metadata_for_config(
            config, force_refresh=refresh_cache
        )
        if not config_result.get("success", False):
            raise ValueError(
                f"Failed to load metadata '{metadata_config_name}': "
                f"{config_result.get('message')}"
            )

        # TODO: fix this type ignore
        return config_result.get("table_name")  # type: ignore

    def _build_join_sql(
        self,
        base_sql: str,
        base_table: str,
        metadata_joins: list[tuple[str, str, list[str]]],
    ) -> str:
        """
        Rewrite SQL to include metadata JOINs.

        :param base_sql: Original SQL query
        :param base_table: Base table name
        :param metadata_joins: List of (metadata_config, metadata_table, join_keys)
        :return: Rewritten SQL with JOINs

        """
        if not metadata_joins:
            return base_sql

        # Extract the FROM clause position
        from_pattern = r"\bFROM\s+" + re.escape(base_table)
        match = re.search(from_pattern, base_sql, re.IGNORECASE)
        if not match:
            # Can't find FROM clause, return original
            self.logger.warning("Could not find FROM clause for automatic join")
            return base_sql

        from_end = match.end()

        # Build JOIN clauses
        join_clauses = []
        for metadata_config, metadata_table, join_keys in metadata_joins:
            # Use USING clause to avoid duplicate join columns in result
            # USING automatically deduplicates the join keys
            join_keys_str = ", ".join(join_keys)
            join_clause = f"\nLEFT JOIN {metadata_table} USING ({join_keys_str})"
            join_clauses.append(join_clause)

        # Insert JOINs after FROM clause
        sql_before = base_sql[:from_end]
        sql_after = base_sql[from_end:]

        # Check if there's already a WHERE/GROUP BY/etc after FROM
        # We need to insert JOINs before those
        insert_keywords = ["WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT"]
        insert_position = len(sql_after)

        for keyword in insert_keywords:
            match = re.search(r"\b" + keyword + r"\b", sql_after, re.IGNORECASE)
            if match and match.start() < insert_position:
                insert_position = match.start()

        final_sql = (
            sql_before
            + "".join(join_clauses)
            + " "
            + sql_after[:insert_position].strip()
            + " "
            + sql_after[insert_position:]
        )

        return final_sql.strip()
