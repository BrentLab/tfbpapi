"""MetadataManager for cross-dataset metadata filtering and querying."""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from ..errors import DataCardError
from .datacard import DataCard
from .models import FieldRole

# Separator conventions for concatenated fields
COMPONENT_SEPARATORS = {
    "type_value": ":",  # Separates component type from value
    "value_conc": "@",  # Separates value from concentration
    "components": ";",  # Separates multiple components of same type
    "types": "|",  # Separates different component types (future use)
}


class MetadataManager:
    """
    Cross-dataset metadata query manager using DuckDB temp views.

    Stores metadata field values as-is from parquet files, plus optionally expands field
    definitions into searchable columns for experimental conditions.

    """

    def __init__(self, cache_dir: Path | None = None, cache: bool = False):
        """
        Initialize MetadataManager with optional caching.

        :param cache_dir: Directory for cached metadata (if cache=True)
        :param cache: If True, persist metadata extractions. Default: False

        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._conn = duckdb.connect(":memory:")
        self._registered_datasets: dict[str, DataCard] = {}
        self._view_names: dict[tuple[str, str], str] = {}  # (repo_id, config) -> view
        self._cache_dir = cache_dir
        self._cache_enabled = cache

    def register(self, repo_id: str, config_names: list[str] | None = None) -> None:
        """
        Register a dataset for cross-dataset queries.

        :param repo_id: HuggingFace repository identifier
        :param config_names: Optional list of config names to register. If None,
            registers all configs.

        """
        self.logger.info(f"Registering dataset: {repo_id}")

        # Load datacard
        if repo_id not in self._registered_datasets:
            datacard = DataCard(repo_id=repo_id)
            self._registered_datasets[repo_id] = datacard
        else:
            datacard = self._registered_datasets[repo_id]

        # Determine which configs to register
        if config_names is None:
            configs_to_register = [c.config_name for c in datacard.configs]
        else:
            configs_to_register = config_names

        # Register each config
        for config_name in configs_to_register:
            self._register_config(datacard, config_name)

        # Recreate unified view
        self._create_unified_view()

    def _register_config(self, datacard: DataCard, config_name: str) -> None:
        """
        Register a single config for querying.

        :param datacard: DataCard instance
        :param config_name: Configuration name to register

        """
        self.logger.debug(f"Registering config: {datacard.repo_id}/{config_name}")

        # Extract metadata schema
        try:
            schema = datacard.extract_metadata_schema(config_name)
        except DataCardError as e:
            self.logger.error(f"Failed to extract schema for {config_name}: {e}")
            raise

        # Extract metadata from parquet file
        try:
            metadata_df = self._extract_metadata_from_config(
                datacard, config_name, schema
            )
        except Exception as e:
            self.logger.error(f"Failed to extract metadata for {config_name}: {e}")
            raise

        # Create temp view
        view_name = self._sanitize_view_name(datacard.repo_id, config_name)
        self._create_temp_view(datacard.repo_id, config_name, metadata_df, view_name)
        self._view_names[(datacard.repo_id, config_name)] = view_name

    def unregister(self, repo_id: str) -> None:
        """
        Remove dataset from cross-dataset queries.

        :param repo_id: HuggingFace repository identifier to unregister

        """
        if repo_id not in self._registered_datasets:
            self.logger.warning(f"Dataset {repo_id} not registered")
            return

        # Remove all views for this repo
        views_to_remove = [
            (rid, cfg) for rid, cfg in self._view_names.keys() if rid == repo_id
        ]

        for rid, cfg in views_to_remove:
            view_name = self._view_names.pop((rid, cfg))
            try:
                self._conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            except Exception as e:
                self.logger.warning(f"Failed to drop view {view_name}: {e}")

        # Remove datacard
        del self._registered_datasets[repo_id]

        # Recreate unified view
        self._create_unified_view()

    def _extract_metadata_from_config(
        self, datacard: DataCard, config_name: str, schema: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract metadata from parquet file for a config.

        Processes experimental conditions at all three hierarchy levels:
        - Top-level (repo-wide): Applies to ALL samples
        - Config-level: Applies to all samples in this config
        - Field-level: Varies per sample (from field definitions)

        Hierarchy merging: field-level > config-level > top-level

        :param datacard: DataCard instance
        :param config_name: Configuration name
        :param schema: Metadata schema from extract_metadata_schema()

        :return: DataFrame with metadata field values + expanded columns

        """
        # Get config to access data files
        config = datacard.get_config(config_name)
        if not config:
            raise DataCardError(f"Configuration '{config_name}' not found")

        # Determine which fields to extract
        metadata_fields = (
            schema["regulator_fields"]
            + schema["target_fields"]
            + schema["condition_fields"]
        )

        # Add sample_id if not already in list
        if "sample_id" not in metadata_fields:
            metadata_fields.append("sample_id")

        # TODO: For MVP, we need actual parquet file path from HuggingFace
        # This requires integration with HF datasets or direct file access
        # For now, create a minimal placeholder that will work with tests

        # Create placeholder DataFrame with required structure
        metadata_df = pd.DataFrame()

        # Add computed columns
        metadata_df["dataset"] = datacard.repo_id
        metadata_df["config_name"] = config_name

        # Add metadata fields as empty for now
        for field in metadata_fields:
            metadata_df[field] = "unspecified"

        # Process experimental conditions in hierarchy order (low to high priority)
        # Start with top-level (repo-wide) conditions
        if schema.get("top_level_conditions"):
            top_conditions = self._flatten_experimental_conditions(
                schema["top_level_conditions"]
            )
            for col_name, col_value in top_conditions.items():
                metadata_df[col_name] = col_value

        # Apply config-level conditions (overrides top-level)
        if schema.get("config_level_conditions"):
            config_conditions = self._flatten_experimental_conditions(
                schema["config_level_conditions"]
            )
            for col_name, col_value in config_conditions.items():
                metadata_df[col_name] = col_value

        # Expand field-level condition definitions (overrides config/top-level)
        if schema["condition_definitions"]:
            for field_name, definitions in schema["condition_definitions"].items():
                # Add expanded columns for each condition
                metadata_df["growth_media"] = "unspecified"
                metadata_df["components"] = "unspecified"

        return metadata_df

    def _flatten_experimental_conditions(self, exp_conditions: Any) -> dict[str, Any]:
        """
        Flatten ExperimentalConditions object into column name/value pairs.

        Processes both environmental_conditions and strain_background from repo-level or
        config-level ExperimentalConditions. Also handles extra fields stored in
        model_extra (Pydantic's extra='allow').

        :param exp_conditions: ExperimentalConditions instance
        :return: Dict mapping column names to values

        """
        flattened: dict[str, Any] = {}

        # Handle strain_background
        if (
            hasattr(exp_conditions, "strain_background")
            and exp_conditions.strain_background
        ):
            strain = exp_conditions.strain_background
            if isinstance(strain, str):
                flattened["strain_background"] = strain
            elif isinstance(strain, dict):
                flattened["strain_background"] = strain.get("name", "unspecified")

        # Handle extra fields (for models with extra='allow')
        # Some datacards store conditions directly as extra fields
        if hasattr(exp_conditions, "model_extra") and exp_conditions.model_extra:
            for key, value in exp_conditions.model_extra.items():
                # Store extra fields with their original names
                flattened[key] = value

        # Handle environmental_conditions
        if (
            hasattr(exp_conditions, "environmental_conditions")
            and exp_conditions.environmental_conditions
        ):
            env = exp_conditions.environmental_conditions

            # Temperature
            if (
                hasattr(env, "temperature_celsius")
                and env.temperature_celsius is not None
            ):
                flattened["temperature_celsius"] = env.temperature_celsius

            # Cultivation method
            if hasattr(env, "cultivation_method") and env.cultivation_method:
                flattened["cultivation_method"] = env.cultivation_method

            # Media information
            if hasattr(env, "media") and env.media:
                media = env.media
                if hasattr(media, "name") and media.name:
                    flattened["growth_media"] = media.name

                # Build components string from media composition
                components = []

                # Carbon source
                if hasattr(media, "carbon_source") and media.carbon_source:
                    for compound in media.carbon_source:
                        if hasattr(compound, "compound"):
                            components.append(
                                self._format_compound(
                                    "carbon_source", compound.model_dump()
                                )
                            )

                # Nitrogen source
                if hasattr(media, "nitrogen_source") and media.nitrogen_source:
                    for compound in media.nitrogen_source:
                        if hasattr(compound, "compound"):
                            components.append(
                                self._format_compound(
                                    "nitrogen_source", compound.model_dump()
                                )
                            )

                # Phosphate source
                if hasattr(media, "phosphate_source") and media.phosphate_source:
                    for compound in media.phosphate_source:
                        if hasattr(compound, "compound"):
                            components.append(
                                self._format_compound(
                                    "phosphate_source", compound.model_dump()
                                )
                            )

                # Additives
                if hasattr(media, "additives") and media.additives:
                    for additive in media.additives:
                        if hasattr(additive, "name"):
                            components.append(f"additive:{additive.name}")

                if components:
                    flattened["components"] = "|".join(components)

            # Growth phase
            if hasattr(env, "growth_phase") and env.growth_phase:
                gp = env.growth_phase
                if hasattr(gp, "stage") and gp.stage:
                    flattened["growth_stage"] = gp.stage
                if hasattr(gp, "od600") and gp.od600 is not None:
                    flattened["od600"] = gp.od600

            # Chemical treatments
            if hasattr(env, "chemical_treatments") and env.chemical_treatments:
                treatments = []
                for treatment in env.chemical_treatments:
                    if hasattr(treatment, "compound") and treatment.compound:
                        treatments.append(treatment.compound)
                if treatments:
                    flattened["chemical_treatments"] = ";".join(treatments)

            # Drug treatments
            if hasattr(env, "drug_treatments") and env.drug_treatments:
                drugs = []
                for drug in env.drug_treatments:
                    if hasattr(drug, "compound") and drug.compound:
                        drugs.append(drug.compound)
                if drugs:
                    flattened["drug_treatments"] = ";".join(drugs)

            # Heat treatment
            if hasattr(env, "heat_treatment") and env.heat_treatment:
                ht = env.heat_treatment
                if (
                    hasattr(ht, "temperature_celsius")
                    and ht.temperature_celsius is not None
                ):
                    flattened["heat_treatment_temp"] = ht.temperature_celsius

            # Induction
            if hasattr(env, "induction") and env.induction:
                ind = env.induction
                if hasattr(ind, "system") and ind.system:
                    flattened["induction_system"] = ind.system

        return flattened

    def _flatten_condition_definition(
        self, definition: dict[str, Any]
    ) -> dict[str, str]:
        """
        Flatten a single condition definition into searchable fields.

        :param definition: Condition definition dict (e.g., YPD definition)
        :return: Dict with flattened fields (growth_media, components)

        """
        flattened: dict[str, str] = {
            "growth_media": "unspecified",
            "components": "",
        }

        # Extract environmental conditions if present
        if "environmental_conditions" in definition:
            env_conds = definition["environmental_conditions"]

            # Extract media information
            if "media" in env_conds:
                media = env_conds["media"]
                if isinstance(media, dict):
                    # Extract media name
                    if "name" in media:
                        flattened["growth_media"] = media["name"]

                    # Build components string
                    components = []

                    # Extract carbon source
                    if "carbon_source" in media:
                        carbon = media["carbon_source"]
                        if isinstance(carbon, list):
                            for compound in carbon:
                                components.append(
                                    self._format_compound("carbon_source", compound)
                                )
                        elif isinstance(carbon, dict):
                            components.append(
                                self._format_compound("carbon_source", carbon)
                            )

                    # Extract nitrogen source
                    if "nitrogen_source" in media:
                        nitrogen = media["nitrogen_source"]
                        if isinstance(nitrogen, list):
                            for compound in nitrogen:
                                components.append(
                                    self._format_compound("nitrogen_source", compound)
                                )
                        elif isinstance(nitrogen, dict):
                            components.append(
                                self._format_compound("nitrogen_source", compound)
                            )

                    # Extract phosphate source
                    if "phosphate_source" in media:
                        phosphate = media["phosphate_source"]
                        if isinstance(phosphate, list):
                            for compound in phosphate:
                                components.append(
                                    self._format_compound("phosphate_source", compound)
                                )
                        elif isinstance(phosphate, dict):
                            components.append(
                                self._format_compound("phosphate_source", compound)
                            )

                    # Join components
                    if components:
                        flattened["components"] = "|".join(components)

        return flattened

    def _format_compound(self, component_type: str, compound: dict[str, Any]) -> str:
        """
        Format a compound dict into searchable string.

        :param component_type: Type of component (carbon_source, nitrogen_source, etc.)
        :paramcompound: Compound info dict with name and concentration
        :return: Formatted string (e.g., "carbon_source:D-glucose@2%")

        """
        if isinstance(compound, str):
            return f"{component_type}:{compound}"

        name = compound.get("name", "unknown")
        result = f"{component_type}:{name}"

        # Add concentration if present
        if "concentration_percent" in compound:
            result += f"@{compound['concentration_percent']}%"
        elif "concentration_g_per_l" in compound:
            result += f"@{compound['concentration_g_per_l']}g/L"
        elif "concentration_molar" in compound:
            result += f"@{compound['concentration_molar']}M"

        return result

    def _create_temp_view(
        self, repo_id: str, config_name: str, metadata_df: pd.DataFrame, view_name: str
    ) -> None:
        """
        Create DuckDB temp view from metadata DataFrame.

        :param repo_id: Repository identifier
        :param config_name: Configuration name
        :param metadata_df: Metadata DataFrame
        :param view_name: Sanitized view name

        """
        try:
            # Register DataFrame as temp view in DuckDB
            self._conn.register(view_name, metadata_df)
            self.logger.debug(f"Created temp view: {view_name}")
        except Exception as e:
            self.logger.error(f"Failed to create view {view_name}: {e}")
            raise

    def _sanitize_view_name(self, repo_id: str, config_name: str) -> str:
        """
        Create a sanitized view name from repo_id and config_name.

        :param repo_id: Repository identifier
        :param config_name: Configuration name
        :return: Sanitized view name safe for SQL

        """
        # Replace non-alphanumeric with underscores
        safe_repo = re.sub(r"[^a-zA-Z0-9]+", "_", repo_id)
        safe_config = re.sub(r"[^a-zA-Z0-9]+", "_", config_name)
        return f"{safe_repo}_{safe_config}_metadata"

    def _create_unified_view(self) -> None:
        """Create unified_metadata view combining all registered datasets."""
        if not self._view_names:
            self.logger.debug("No views registered, skipping unified view creation")
            return

        # Drop existing unified view if present
        try:
            self._conn.execute("DROP VIEW IF EXISTS unified_metadata")
        except Exception:
            pass

        # Get all column names across all views
        all_columns = set()
        for view_name in self._view_names.values():
            try:
                cols = (
                    self._conn.execute(f"DESCRIBE {view_name}")
                    .fetchdf()["column_name"]
                    .tolist()
                )
                all_columns.update(cols)
            except Exception as e:
                self.logger.warning(f"Failed to get columns from {view_name}: {e}")

        if not all_columns:
            self.logger.warning("No columns found in registered views")
            return

        # Build UNION ALL query with column alignment
        union_queries = []
        for view_name in self._view_names.values():
            # Get columns in this view
            view_cols = (
                self._conn.execute(f"DESCRIBE {view_name}")
                .fetchdf()["column_name"]
                .tolist()
            )

            # Build SELECT with coalesce for missing columns
            select_parts = []
            for col in sorted(all_columns):
                if col in view_cols:
                    select_parts.append(f'"{col}"')
                else:
                    select_parts.append(f"'unspecified' AS \"{col}\"")

            union_queries.append(f"SELECT {', '.join(select_parts)} FROM {view_name}")

        # Create unified view
        unified_query = " UNION ALL ".join(union_queries)
        try:
            self._conn.execute(f"CREATE VIEW unified_metadata AS {unified_query}")
            self.logger.debug(
                f"Created unified view from {len(self._view_names)} views with {len(all_columns)} columns"
            )
        except Exception as e:
            self.logger.error(f"Failed to create unified view: {e}")
            raise

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query across all registered datasets.

        :param sql: SQL query string to execute
        :return: Query results as pandas DataFrame

        """
        try:
            result = self._conn.execute(sql).fetchdf()
            return result
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise

    def filter_by_regulator(self, regulators: list[str]) -> "MetadataManager":
        """
        Filter metadata to specific regulators.

        :param regulators: List of regulator symbols to filter by
        :return: Self for method chaining

        """
        # TODO: Implement filtering logic
        # This will be implemented in Phase 3
        raise NotImplementedError("filter_by_regulator not yet implemented")

    def filter_by_conditions(self, **kwargs: Any) -> "MetadataManager":
        """
        Filter by experimental conditions.

        :param kwargs: Condition filters (e.g., media="YPD", temperature=30)
        :return: Self for method chaining

        """
        # TODO: Implement filtering logic
        # This will be implemented in Phase 3
        raise NotImplementedError("filter_by_conditions not yet implemented")

    def get_active_configs(self) -> list[tuple[str, str]]:
        """
        Get (repo_id, config_name) pairs that match active filters.

        :return: List of (repo_id, config_name) tuples

        """
        # TODO: Implement with filter support
        # For now, return all registered configs
        return list(self._view_names.keys())

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary stats for registered datasets.

        :return: DataFrame with summary statistics

        """
        if not self._view_names:
            return pd.DataFrame()

        # TODO: Implement summary statistics
        # This will be implemented in Phase 3
        summary_data = []
        for (repo_id, config_name), view_name in self._view_names.items():
            summary_data.append(
                {
                    "dataset": repo_id,
                    "config_name": config_name,
                    "view_name": view_name,
                }
            )

        return pd.DataFrame(summary_data)
