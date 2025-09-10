import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class IncrementalAnalysisDB:
    """
    Class for managing incremental analysis results in DuckDB.

    Supports appending new results, updating existing ones, and maintaining analysis
    metadata for tracking what's been computed.

    """

    def __init__(self, db_path: str):
        """
        Initialize connection to persistent DuckDB database.

        :param db_path: Path to the DuckDB database file

        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self.logger = logging.getLogger(__name__)

        # Create metadata table to track analyses
        self._ensure_metadata_table()

    def _ensure_metadata_table(self):
        """Create metadata table if it doesn't exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_metadata (
                table_name VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_records INTEGER,
                analysis_type VARCHAR,
                parameters JSON,
                description TEXT
            )
        """
        )

    def append_results(
        self,
        new_results: pd.DataFrame,
        table_name: str,
        analysis_type: str = "response_rate",
        parameters: dict | None = None,
        description: str | None = None,
        deduplicate_on: list[str] | None = None,
    ) -> int:
        """
        Append new analysis results to an existing table.

        :param new_results: DataFrame with new results to append
        :param table_name: Name of the target table
        :param analysis_type: Type of analysis for metadata
        :param parameters: Parameters used in the analysis
        :param description: Description of the analysis
        :param deduplicate_on: Column names to deduplicate on
        :return: Number of records added

        """
        if new_results.empty:
            self._update_metadata(table_name, 0, analysis_type, parameters, description)
            return 0

        # Handle deduplication if specified
        if deduplicate_on and self.table_exists(table_name):
            existing_data = self.get_results(table_name)
            if not existing_data.empty:
                # Remove duplicates based on specified columns
                merged = pd.merge(
                    new_results,
                    existing_data[deduplicate_on],
                    on=deduplicate_on,
                    how="left",
                    indicator=True,
                )
                new_results = merged[merged["_merge"] == "left_only"].drop(
                    "_merge", axis=1
                )

        # Insert new data
        if not new_results.empty:
            self.conn.register("new_data", new_results)
            if self.table_exists(table_name):
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM new_data")
            else:
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM new_data"
                )
            self.conn.unregister("new_data")

        records_added = len(new_results)
        self._update_metadata(
            table_name, records_added, analysis_type, parameters, description
        )

        return records_added

    def update_results(
        self, updated_data: pd.DataFrame, table_name: str, key_columns: list[str]
    ) -> int:
        """
        Update existing records in a table.

        :param updated_data: DataFrame with updated values
        :param table_name: Name of the target table
        :param key_columns: Columns to match records on
        :return: Number of records updated

        """
        if not self.table_exists(table_name) or updated_data.empty:
            return 0

        records_updated = 0
        self.conn.register("update_data", updated_data)

        # Build SET clause for non-key columns
        non_key_columns = [
            col for col in updated_data.columns if col not in key_columns
        ]
        set_clause = ", ".join(
            [f"{col} = update_data.{col}" for col in non_key_columns]
        )

        # Build WHERE clause for key columns
        where_clause = " AND ".join(
            [f"{table_name}.{col} = update_data.{col}" for col in key_columns]
        )

        update_query = f"""
            UPDATE {table_name}
            SET {set_clause}
            FROM update_data
            WHERE {where_clause}
        """

        self.conn.execute(update_query)
        records_updated = len(updated_data)

        self.conn.unregister("update_data")
        self._update_metadata_timestamp(table_name)

        return records_updated

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        :param sql: SQL query to execute
        :return: DataFrame with query results

        """
        return self.conn.execute(sql).fetchdf()

    def get_results(
        self,
        table_name: str,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve results from a table.

        :param table_name: Name of the table to query
        :param filters: Optional filters to apply
        :param limit: Optional limit on number of records
        :return: DataFrame with results

        """
        if not self.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")

        query = f"SELECT * FROM {table_name}"

        if filters:
            where_conditions = []
            for column, values in filters.items():
                if isinstance(values, list):
                    values_str = ", ".join(
                        [f"'{v}'" if isinstance(v, str) else str(v) for v in values]
                    )
                    where_conditions.append(f"{column} IN ({values_str})")
                else:
                    if isinstance(values, str):
                        where_conditions.append(f"{column} = '{values}'")
                    else:
                        where_conditions.append(f"{column} = {values}")

            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).fetchdf()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = self.conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_name = ? AND table_schema = 'main'
        """,
            [table_name],
        ).fetchall()
        return len(result) > 0

    def drop_table(self, table_name: str) -> None:
        """Drop a table and its metadata."""
        if self.table_exists(table_name):
            self.conn.execute(f"DROP TABLE {table_name}")
            self.conn.execute(
                "DELETE FROM analysis_metadata WHERE table_name = ?", [table_name]
            )

    def get_table_info(self, table_name: str) -> dict[str, Any]:
        """Get metadata information about a table."""
        if not self.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")

        result = self.conn.execute(
            """
            SELECT * FROM analysis_metadata WHERE table_name = ?
        """,
            [table_name],
        ).fetchdf()

        if result.empty:
            raise ValueError(f"No metadata found for table {table_name}")

        return result.iloc[0].to_dict()

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        result = self.conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main'
        """
        ).fetchall()
        return [row[0] for row in result]

    def get_table_schema(self, table_name: str) -> list[dict[str, str]]:
        """Get schema information for a table."""
        if not self.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")

        result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        # TODO: fix the mypy ignore/typing
        return [
            {
                "column_name": row[0],
                "column_type": row[1],
                "null": row[2],
                "key": row[3] if len(row) > 3 else None,  # type: ignore
                "default": row[4] if len(row) > 4 else None,  # type: ignore
                "extra": row[5] if len(row) > 5 else None,  # type: ignore
            }
            for row in result
        ]

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "conn"):
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _update_metadata(
        self,
        table_name: str,
        records_added: int,
        analysis_type: str,
        parameters: dict | None,
        description: str | None,
    ) -> None:
        """Update or insert metadata for a table."""
        import json

        # Check if metadata exists
        existing = self.conn.execute(
            """
            SELECT total_records FROM analysis_metadata WHERE table_name = ?
        """,
            [table_name],
        ).fetchall()

        if existing:
            # Update existing metadata
            new_total = existing[0][0] + records_added
            self.conn.execute(
                """
                UPDATE analysis_metadata
                SET last_updated = CURRENT_TIMESTAMP, total_records = ?
                WHERE table_name = ?
            """,
                [new_total, table_name],
            )
        else:
            # Insert new metadata
            self.conn.execute(
                """
                INSERT INTO analysis_metadata
                (table_name, total_records, analysis_type, parameters, description)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    table_name,
                    records_added,
                    analysis_type,
                    json.dumps(parameters) if parameters else None,
                    description,
                ],
            )

    def _update_metadata_timestamp(self, table_name: str) -> None:
        """Update the last_updated timestamp for a table."""
        self.conn.execute(
            """
            UPDATE analysis_metadata
            SET last_updated = CURRENT_TIMESTAMP
            WHERE table_name = ?
        """,
            [table_name],
        )
