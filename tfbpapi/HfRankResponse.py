import logging

import duckdb
import pandas as pd

from .IncrementalAnalysisDB import IncrementalAnalysisDB


class HfRankResponse:
    """
    A class to provide an API to compute and analyze "rank response", which is defined
    as the cumulative number of responsive targets (e.g., genes) binned by their binding
    rank score for each regulator sample pair of binding and perturbation response data.

    Handles multiple dataset comparisons and stores all results in a shared database.

    """

    def __init__(self, db: IncrementalAnalysisDB):
        """
        Initialize RankResponse analyzer with database connection.

        :param db: IncrementalAnalysisDB instance for storing results

        """
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute(
        self,
        ranking_api,
        response_api,
        ranking_table: str,
        response_table: str,
        ranking_score_column: str,
        response_column: str,
        comparison_id: str | None = None,
        regulator_column: str = "regulator_locus_tag",
        target_column: str = "target_locus_tag",
        bin_size: int = 5,
        force_recompute: bool = False,
        responsive_condition: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute rank response for a specific dataset comparison.

        :param ranking_api: API instance for ranking/binding data
        :param response_api: API instance for response/perturbation data
        :param ranking_table: Name of ranking table in ranking_api
        :param response_table: Name of response table in response_api
        :param ranking_score_column: Column name for ranking scores
        :param response_column: Column name for response values
        :param comparison_id: Unique identifier for this comparison (auto-generated if
            None)
        :param regulator_column: Column name for regulator identifiers
        :param target_column: Column name for target identifiers
        :param bin_size: Size of ranking bins
        :param force_recompute: Whether to recompute existing results
        :param responsive_condition: SQL condition to define responsive (default: IS NOT
            NULL)
        :return: DataFrame with computed results

        """
        # Generate comparison ID if not provided
        if comparison_id is None:
            comparison_id = f"{ranking_table}_vs_{response_table}"

        table_name = f"rank_response_{comparison_id}"

        # Get all regulators from ranking data
        ranking_api._ensure_dataset_loaded(ranking_table)
        all_regulators = ranking_api.query(
            f"SELECT DISTINCT {regulator_column} FROM {ranking_table}"
        )[regulator_column].tolist()

        # Check which regulators already have results
        if not force_recompute and self.db.table_exists(table_name):
            existing_regulators = set(
                self.db.query(f"SELECT DISTINCT {regulator_column} FROM {table_name}")[
                    regulator_column
                ].tolist()
            )

            new_regulators = [
                reg for reg in all_regulators if reg not in existing_regulators
            ]
            self.logger.info(
                f"Found {len(existing_regulators)} existing regulators, "
                f"{len(new_regulators)} new ones"
            )
        else:
            new_regulators = all_regulators
            self.logger.info(f"Computing analysis for {len(new_regulators)} regulators")

        if not new_regulators:
            self.logger.info("No new regulators to analyze")
            return self.db.query(
                f"SELECT * FROM {table_name} ORDER BY "
                f"{regulator_column}, {target_column}"
            )

        # Apply filters to focus on new regulators
        # For analytical queries, escape and format the values safely
        escaped_values = [
            f"'{reg.replace(chr(39), chr(39)+chr(39))}'" for reg in new_regulators
        ]
        regulator_filter = f"{regulator_column} IN ({', '.join(escaped_values)})"

        # Temporarily add filters for new regulators only
        original_ranking_filter = ranking_api.get_table_filter(ranking_table)
        original_response_filter = response_api.get_table_filter(response_table)

        new_ranking_filter = regulator_filter
        new_response_filter = regulator_filter

        if original_ranking_filter:
            new_ranking_filter = f"({original_ranking_filter}) AND ({regulator_filter})"
        if original_response_filter:
            new_response_filter = (
                f"({original_response_filter}) AND ({regulator_filter})"
            )

        ranking_api.set_table_filter(ranking_table, new_ranking_filter)
        response_api.set_table_filter(response_table, new_response_filter)

        try:
            # Load filtered data from both APIs
            ranking_api._ensure_dataset_loaded(ranking_table)
            response_api._ensure_dataset_loaded(response_table)

            ranking_conn = ranking_api._duckdb_conn
            response_conn = response_api._duckdb_conn

            # Create temporary connection for analysis
            temp_conn = duckdb.connect(":memory:")

            try:
                # Get filtered data
                ranking_df = ranking_conn.execute(
                    f"SELECT * FROM {ranking_table}"
                ).fetchdf()
                response_df = response_conn.execute(
                    f"SELECT * FROM {response_table}"
                ).fetchdf()

                temp_conn.register("ranking_data", ranking_df)
                temp_conn.register("response_data", response_df)

                # Execute intermediate analysis SQL
                # Set default responsive condition if not provided
                if responsive_condition is None:
                    responsive_condition = f"b.{response_column} IS NOT NULL"
                else:
                    # Replace column references in the condition
                    responsive_condition = responsive_condition.replace(
                        response_column, f"b.{response_column}"
                    )

                intermediate_sql = f"""
                WITH binned_data AS (
                    SELECT
                        a.{regulator_column},
                        a.{target_column},
                        a.{ranking_score_column},
                        b.{response_column},
                        CASE WHEN {responsive_condition} THEN 1 ELSE 0 END
                        AS responsive,
                        CEILING(ROW_NUMBER() OVER (
                            PARTITION BY a.{regulator_column}
                            ORDER BY a.{regulator_column}, a.{target_column}
                        ) / {bin_size}.0) * {bin_size} AS bin_label
                    FROM ranking_data AS a
                    LEFT JOIN response_data AS b
                    ON a.{regulator_column} = b.{regulator_column}
                    AND a.{target_column} = b.{target_column}
                )
                SELECT
                    {regulator_column},
                    {target_column},
                    {ranking_score_column},
                    {response_column},
                    responsive,
                    bin_label,
                    SUM(responsive) OVER (
                        PARTITION BY {regulator_column}
                        ORDER BY bin_label
                        RANGE UNBOUNDED PRECEDING
                    ) AS cumulative_responsive
                FROM binned_data
                ORDER BY {regulator_column}, bin_label, {target_column}
                """

                new_results = temp_conn.execute(intermediate_sql).fetchdf()

            finally:
                temp_conn.close()

            # Save new intermediate results
            if len(new_results) > 0:
                self.db.append_results(
                    new_results,
                    table_name,
                    analysis_type="response_rate_intermediate",
                    parameters={
                        "ranking_table": ranking_table,
                        "response_table": response_table,
                        "ranking_score_column": ranking_score_column,
                        "response_column": response_column,
                        "bin_size": bin_size,
                        "result_type": "intermediate",
                    },
                    description=(
                        f"Added intermediate data for {len(new_regulators)} "
                        "new regulators"
                    ),
                    deduplicate_on=[regulator_column, target_column],
                )

                self.logger.info(
                    f"Saved {len(new_results)} intermediate records to database"
                )

            # Return complete results from database
            return self.db.query(
                f"SELECT * FROM {table_name} ORDER BY {regulator_column}, bin_label, "
                f"{target_column}"
            )

        finally:
            # Restore original filters
            if original_ranking_filter:
                ranking_api.set_table_filter(ranking_table, original_ranking_filter)
            else:
                ranking_api.remove_table_filter(ranking_table)

            if original_response_filter:
                response_api.set_table_filter(response_table, original_response_filter)
            else:
                response_api.remove_table_filter(response_table)

    def get_comparisons(self) -> list[str]:
        """
        Get list of all computed comparisons.

        :return: List of comparison identifiers

        """
        tables = self.db.list_tables()
        rank_response_tables = [
            table
            for table in tables
            if table.startswith("rank_response_") and table != "rank_response_metadata"
        ]
        return [table.replace("rank_response_", "") for table in rank_response_tables]

    def get_bin_summary(
        self,
        comparison_id: str,
        regulator_column: str = "regulator_locus_tag",
        bin_size: int = 5,
        regulators_filter: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Generate bin-level summary for a specific comparison.

        :param comparison_id: Identifier for the comparison to summarize
        :param regulator_column: Column name for regulator identifiers
        :param bin_size: Bin size used in analysis
        :param regulators_filter: Optional list of regulators to include
        :return: DataFrame with bin summary results

        """
        intermediate_table_name = f"rank_response_{comparison_id}"

        if not self.db.table_exists(intermediate_table_name):
            raise ValueError(
                f"Intermediate table '{intermediate_table_name}' does not exist. "
                "Run compute() first."
            )

        # Build WHERE clause for regulator filter
        where_clause = ""
        if regulators_filter:
            # For analytical queries, escape and format the values safely
            escaped_values = [
                f"'{reg.replace(chr(39), chr(39)+chr(39))}'"
                for reg in regulators_filter
            ]
            where_clause = f"WHERE {regulator_column} IN ({', '.join(escaped_values)})"

        # Generate summary from intermediate data
        summary_sql = f"""
        SELECT
            {regulator_column},
            bin_label,
            COUNT(*) as records_in_bin,
            SUM(responsive) as responsive_in_bin,
            MAX(cumulative_responsive) as cumulative_responsive,
            MAX(cumulative_responsive) / bin_label as response_rate
        FROM {intermediate_table_name}
        {where_clause}
        GROUP BY {regulator_column}, bin_label
        ORDER BY {regulator_column}, bin_label
        """

        summary_results = self.db.query(summary_sql)

        self.logger.info(
            f"Generated summary for {len(summary_results)} regulator-bin combinations"
        )
        return summary_results

    def get_regulator_summary(
        self,
        comparison_id: str,
        regulator_column: str = "regulator_locus_tag",
        max_bin_label: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate regulator-level performance summary for a comparison.

        :param comparison_id: Identifier for the comparison
        :param regulator_column: Column name for regulator identifiers
        :param max_bin_label: Maximum bin label to consider (e.g., 20 for top 20
            targets)
        :return: DataFrame with regulator-level summary statistics

        """
        intermediate_table_name = f"rank_response_{comparison_id}"

        if not self.db.table_exists(intermediate_table_name):
            raise ValueError(
                f"Intermediate table '{intermediate_table_name}' does not exist."
            )

        where_clause = ""
        if max_bin_label:
            where_clause = f"WHERE bin_label <= {max_bin_label}"

        regulator_summary_sql = f"""
        SELECT
            {regulator_column},
            COUNT(*) as total_targets,
            SUM(responsive) as total_responsive,
            COUNT(DISTINCT bin_label) as num_bins,
            MAX(cumulative_responsive) as max_cumulative_responsive,
            MAX(bin_label) as max_bin_label,
            MAX(cumulative_responsive) / MAX(bin_label)
            as overall_response_rate,
            AVG(CASE WHEN bin_label <= 5 THEN responsive ELSE NULL END)
            as top5_response_rate,
            AVG(CASE WHEN bin_label <= 10 THEN responsive ELSE NULL END)
            as top10_response_rate,
            AVG(CASE WHEN bin_label <= 20 THEN responsive ELSE NULL END)
            as top20_response_rate
        FROM {intermediate_table_name}
        {where_clause}
        GROUP BY {regulator_column}
        ORDER BY overall_response_rate DESC
        """

        return self.db.query(regulator_summary_sql)

    def summarize(
        self,
        comparison_id: str,
        summary_type: str = "bin",
        regulator_column: str = "regulator_locus_tag",
        bin_size: int = 5,
        regulators_filter: list[str] | None = None,
        max_bin_label: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate summary for a specific comparison.

        :param comparison_id: Identifier for the comparison to summarize
        :param summary_type: Type of summary ('bin' or 'regulator')
        :param regulator_column: Column name for regulator identifiers
        :param bin_size: Bin size used in analysis
        :param regulators_filter: Optional list of regulators to include
        :param max_bin_label: Maximum bin label to consider (for regulator summaries)
        :return: DataFrame with summary results

        """
        if summary_type == "bin":
            return self.get_bin_summary(
                comparison_id=comparison_id,
                regulator_column=regulator_column,
                bin_size=bin_size,
                regulators_filter=regulators_filter,
            )
        elif summary_type == "regulator":
            return self.get_regulator_summary(
                comparison_id=comparison_id,
                regulator_column=regulator_column,
                max_bin_label=max_bin_label,
            )
        else:
            raise ValueError(f"Unknown summary type: {summary_type}")

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute custom SQL query on the database.

        :param sql: SQL query to execute
        :return: DataFrame with query results

        """
        return self.db.query(sql)

    def get_comparison_data(
        self,
        comparison_id: str,
        regulator_filter: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Get raw data for a specific comparison.

        :param comparison_id: Identifier for the comparison
        :param regulator_filter: Optional list of regulators to filter
        :param limit: Optional limit on number of records
        :return: DataFrame with raw comparison data

        """
        table_name = f"rank_response_{comparison_id}"

        if not self.db.table_exists(table_name):
            raise ValueError(f"No results found for comparison '{comparison_id}'")

        filters = {}
        if regulator_filter:
            filters["regulator_locus_tag"] = regulator_filter

        return self.db.get_results(table_name, filters=filters, limit=limit)

    def compare_across_datasets(
        self,
        comparison_ids: list[str],
        regulator_column: str = "regulator_locus_tag",
        metric_columns: list[str] = ["overall_response_rate", "top10_response_rate"],
    ) -> pd.DataFrame:
        """
        Compare regulator performance across multiple dataset comparisons.

        :param comparison_ids: List of comparison identifiers to compare
        :param regulator_column: Column name for regulator identifiers
        :param metric_columns: Performance metrics to compare
        :return: DataFrame with cross-comparison results

        """
        comparison_data = []

        for comp_id in comparison_ids:
            summary = self.summarize(comp_id, summary_type="regulator")
            summary["comparison_id"] = comp_id
            comparison_data.append(
                summary[[regulator_column, "comparison_id"] + metric_columns]
            )

        if not comparison_data:
            return pd.DataFrame()

        # Combine all comparisons
        combined = pd.concat(comparison_data, ignore_index=True)

        # Pivot to have comparisons as columns
        result_dfs = []
        for metric in metric_columns:
            pivot = combined.pivot(
                index=regulator_column, columns="comparison_id", values=metric
            )
            pivot.columns = [f"{metric}_{col}" for col in pivot.columns]
            result_dfs.append(pivot)

        if len(result_dfs) == 1:
            return result_dfs[0].reset_index()
        else:
            final_result = result_dfs[0]
            for df in result_dfs[1:]:
                final_result = final_result.join(df)
            return final_result.reset_index()
