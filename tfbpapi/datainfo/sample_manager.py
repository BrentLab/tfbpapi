"""
Node-based sample representation for flexible filtering across heterogeneous datasets.

This module provides a lightweight, NoSQL-inspired approach to managing samples from
multiple datasets with varying experimental condition structures. Each sample is
represented as a node with flattened properties, enabling flexible filtering across
datasets with different metadata schemas.

Key Components:
- SampleNode: Represents a single sample with flattened properties
- SampleNodeCollection: In-memory storage with efficient indexing
- ActiveSet: Filtered collection of samples supporting set operations
- SampleFilter: MongoDB-style query language for filtering
- ConditionFlattener: Handles heterogeneous experimental condition structures
- SampleManager: Main API for loading, filtering, and managing samples

Example Usage:
    >>> manager = SampleManager()
    >>> manager.load_from_datacard("BrentLab/harbison_2004", "harbison_2004")
    >>> active = manager.filter_all({"carbon_source": {"$contains": "glucose"}})
    >>> print(f"Found {len(active)} glucose-grown samples")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import duckdb
import pandas as pd


def get_nested_value(data: dict, path: str) -> Any:
    """
    Navigate nested dict using dot notation.

    Handles missing intermediate keys gracefully by returning None.

    :param data: Dictionary to navigate
    :param path: Dot-separated path (e.g., "environmental_conditions.media.name")
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


def flatten_compound_list(compounds: list[dict] | str | None) -> str:
    """
    Flatten compound list to comma-separated string.

    Handles various representations:
    - List of dicts: Extract compound names
    - String "unspecified": Return as-is
    - None or empty list: Return "unspecified"

    :param compounds: Compound list or string
    :return: Comma-separated compound names or "unspecified"
    """
    if compounds is None or compounds == "unspecified":
        return "unspecified"

    if isinstance(compounds, str):
        return compounds

    if isinstance(compounds, list):
        if not compounds:
            return "unspecified"
        compound_names = [
            c.get("compound", "") for c in compounds if isinstance(c, dict)
        ]
        return ", ".join(compound_names) if compound_names else "unspecified"

    return "unspecified"


@dataclass
class SampleNode:
    """
    Represents a single sample with flattened experimental condition metadata.

    A sample is uniquely identified by (repo_id, config_name, sample_id) and contains
    flattened properties from the 3-level experimental conditions hierarchy plus
    selected metadata field values.

    Attributes:
        sample_id: Unique identifier within config
        repo_id: Dataset repository (e.g., "BrentLab/harbison_2004")
        config_name: Configuration name (e.g., "harbison_2004")
        properties: Flattened experimental condition properties
        metadata_fields: Selected data fields (regulator, target, etc.)
        property_sources: Tracks which level each property came from (repo/config/field/missing)
    """

    sample_id: str
    repo_id: str
    config_name: str
    properties: dict[str, Any] = field(default_factory=dict)
    metadata_fields: dict[str, Any] = field(default_factory=dict)
    property_sources: dict[str, str] = field(default_factory=dict)

    def global_id(self) -> str:
        """
        Generate unique identifier across all datasets.

        Format: {repo_id}:{config_name}:{sample_id}

        :return: Global sample ID
        """
        return f"{self.repo_id}:{self.config_name}:{self.sample_id}"

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get property value with default fallback.

        :param key: Property name
        :param default: Default value if property not found
        :return: Property value or default
        """
        return self.properties.get(key, default)

    def __repr__(self) -> str:
        """String representation showing global ID and property count."""
        return f"SampleNode({self.global_id()}, {len(self.properties)} properties)"


class ConditionFlattener:
    """
    Flattens experimental conditions from 3-level hierarchy into node properties.

    Handles heterogeneity in experimental condition structures across datasets:
    - Variable nesting depths
    - Missing/optional fields
    - Compound lists
    - Multi-level hierarchical overrides

    Resolution order: repo-level -> config-level -> field-level (field overrides all)

    This flattener dynamically discovers all properties without hardcoded schemas.
    It recursively flattens nested structures using dot notation for keys.
    """

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """
        Recursively flatten nested dict using dot notation.

        :param d: Dictionary to flatten
        :param parent_key: Parent key for recursion
        :param sep: Separator for nested keys
        :return: Flattened dict with dot-notation keys
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                # Recursively flatten nested dicts
                items.extend(
                    ConditionFlattener._flatten_dict(v, new_key, sep=sep).items()
                )
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Handle list of dicts (like compound lists)
                # Store both flattened string and structured version
                items.append((new_key, flatten_compound_list(v)))
                items.append((f"_{new_key}_structured", v))
            else:
                # Store primitive values as-is
                items.append((new_key, v))

        return dict(items)

    @classmethod
    def flatten_conditions(
        cls,
        repo_conditions: dict | None,
        config_conditions: dict | None,
        field_conditions: dict | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """
        Flatten 3-level hierarchy into properties dict.

        Dynamically discovers and flattens all properties without hardcoded schemas.
        Nested structures are flattened using dot notation (e.g., "media.carbon_source").

        :param repo_conditions: Top-level experimental_conditions
        :param config_conditions: Config-level experimental_conditions
        :param field_conditions: Field-level condition definition
        :return: (properties, property_sources) tuple
        """
        properties = {}
        sources = {}

        # Flatten each level
        levels = [
            (repo_conditions, "repo"),
            (config_conditions, "config"),
            (field_conditions, "field"),
        ]

        for conditions, level_name in levels:
            if conditions is None:
                continue

            # Flatten this level
            flattened = cls._flatten_dict(conditions)

            # Merge into properties (later levels override earlier)
            for key, value in flattened.items():
                properties[key] = value
                sources[key] = level_name

        return properties, sources


class SampleNodeCollection:
    """
    In-memory collection of sample nodes with efficient indexing.

    Storage strategy:
    - Primary index: {(repo_id, config_name): {sample_id: SampleNode}}
    - Enables fast dataset-level and cross-dataset operations
    - Memory-efficient for typical workloads (1K-100K samples)

    Attributes:
        _nodes: Two-level dict storing all nodes
    """

    def __init__(self):
        """Initialize empty collection."""
        self._nodes: dict[tuple[str, str], dict[str, SampleNode]] = {}

    def add_node(self, node: SampleNode):
        """
        Add sample node to collection.

        :param node: SampleNode to add
        """
        key = (node.repo_id, node.config_name)
        if key not in self._nodes:
            self._nodes[key] = {}
        self._nodes[key][node.sample_id] = node

    def get_node(
        self, repo_id: str, config_name: str, sample_id: str
    ) -> SampleNode | None:
        """
        Get specific node.

        :param repo_id: Repository ID
        :param config_name: Config name
        :param sample_id: Sample ID
        :return: SampleNode or None if not found
        """
        key = (repo_id, config_name)
        if key not in self._nodes:
            return None
        return self._nodes[key].get(sample_id)

    def get_node_by_global_id(self, global_id: str) -> SampleNode | None:
        """
        Get node by global ID.

        :param global_id: Global ID in format {repo_id}:{config_name}:{sample_id}
        :return: SampleNode or None if not found
        """
        parts = global_id.split(":", 2)
        if len(parts) != 3:
            return None
        return self.get_node(parts[0], parts[1], parts[2])

    def iter_dataset_nodes(
        self, repo_id: str, config_name: str
    ) -> Iterator[SampleNode]:
        """
        Iterate over nodes in specific dataset.

        :param repo_id: Repository ID
        :param config_name: Config name
        :return: Iterator over SampleNodes
        """
        key = (repo_id, config_name)
        if key in self._nodes:
            yield from self._nodes[key].values()

    def iter_all_nodes(self) -> Iterator[SampleNode]:
        """
        Iterate over all nodes in collection.

        :return: Iterator over all SampleNodes
        """
        for dataset_nodes in self._nodes.values():
            yield from dataset_nodes.values()

    def get_dataset_keys(self) -> list[tuple[str, str]]:
        """
        Get list of loaded (repo_id, config_name) pairs.

        :return: List of dataset keys
        """
        return list(self._nodes.keys())

    def count_total_nodes(self) -> int:
        """
        Count total nodes in collection.

        :return: Total number of nodes
        """
        return sum(len(nodes) for nodes in self._nodes.values())

    def count_dataset_nodes(self, repo_id: str, config_name: str) -> int:
        """
        Count nodes in specific dataset.

        :param repo_id: Repository ID
        :param config_name: Config name
        :return: Number of nodes in dataset
        """
        key = (repo_id, config_name)
        return len(self._nodes.get(key, {}))


@dataclass
class ActiveSet:
    """
    Represents a collection of active samples with provenance tracking.

    ActiveSet supports set operations (union, intersection, difference) and
    maintains metadata about how the set was created. This enables building
    complex filters incrementally and tracking analysis provenance.

    Attributes:
        sample_ids: Set of global sample IDs
        name: Optional name for this set
        description: Optional description
        source_filter: Filter query that created this set
        created_at: Timestamp
        parent_set: ID of parent set (for tracking lineage)
    """

    sample_ids: set[str] = field(default_factory=set)
    name: str | None = None
    description: str | None = None
    source_filter: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)
    parent_set: str | None = None

    def __len__(self) -> int:
        """Return number of samples in set."""
        return len(self.sample_ids)

    def union(self, other: ActiveSet, name: str | None = None) -> ActiveSet:
        """
        Create new set with samples from both sets.

        :param other: Another ActiveSet
        :param name: Optional name for new set
        :return: New ActiveSet with union
        """
        return ActiveSet(
            sample_ids=self.sample_ids | other.sample_ids,
            name=name or f"{self.name}_union_{other.name}",
            description=f"Union of {self.name} and {other.name}",
        )

    def intersection(self, other: ActiveSet, name: str | None = None) -> ActiveSet:
        """
        Create new set with samples in both sets.

        :param other: Another ActiveSet
        :param name: Optional name for new set
        :return: New ActiveSet with intersection
        """
        return ActiveSet(
            sample_ids=self.sample_ids & other.sample_ids,
            name=name or f"{self.name}_intersect_{other.name}",
            description=f"Intersection of {self.name} and {other.name}",
        )

    def difference(self, other: ActiveSet, name: str | None = None) -> ActiveSet:
        """
        Create new set with samples in this set but not other.

        :param other: Another ActiveSet
        :param name: Optional name for new set
        :return: New ActiveSet with difference
        """
        return ActiveSet(
            sample_ids=self.sample_ids - other.sample_ids,
            name=name or f"{self.name}_minus_{other.name}",
            description=f"Samples in {self.name} but not {other.name}",
        )

    def to_sample_ids(self) -> list[str]:
        """
        Export as list of global sample IDs.

        :return: Sorted list of global IDs
        """
        return sorted(self.sample_ids)

    def __repr__(self) -> str:
        """String representation showing name and size."""
        return f"ActiveSet(name={self.name}, size={len(self)})"


class SampleFilter:
    """
    Evaluate filter expressions against sample nodes.

    Implements MongoDB-style query language for filtering heterogeneous sample data.

    Supported operators:
    - $eq, $ne: Equality/inequality (default is $eq)
    - $gt, $gte, $lt, $lte: Numeric comparisons
    - $in, $nin: List membership
    - $contains: String/list containment (case-insensitive)
    - $exists: Field presence check
    - $and, $or: Logical operators

    Example queries:
        {"temperature_celsius": 30}  # Simple equality
        {"carbon_source": {"$contains": "glucose"}}  # Contains check
        {"$and": [{"temp": {"$gte": 25}}, {"temp": {"$lte": 35}}]}  # Range
    """

    @staticmethod
    def _matches_operator(value: Any, operator: str, target: Any) -> bool:
        """
        Check if value matches operator condition.

        :param value: Actual property value
        :param operator: Operator string (e.g., "$eq", "$contains")
        :param target: Target value to compare against
        :return: True if condition matches
        """
        # Handle None values
        if value is None:
            if operator == "$exists":
                return not target  # $exists: false matches None
            return operator == "$eq" and target is None

        # Equality operators
        if operator == "$eq":
            return value == target
        if operator == "$ne":
            return value != target

        # Numeric comparisons
        if operator in ["$gt", "$gte", "$lt", "$lte"]:
            try:
                value_num = (
                    float(value) if not isinstance(value, (int, float)) else value
                )
                target_num = (
                    float(target) if not isinstance(target, (int, float)) else target
                )
                if operator == "$gt":
                    return value_num > target_num
                if operator == "$gte":
                    return value_num >= target_num
                if operator == "$lt":
                    return value_num < target_num
                if operator == "$lte":
                    return value_num <= target_num
            except (ValueError, TypeError):
                return False

        # List membership
        if operator == "$in":
            return value in target if isinstance(target, (list, set, tuple)) else False
        if operator == "$nin":
            return (
                value not in target if isinstance(target, (list, set, tuple)) else True
            )

        # Contains check (case-insensitive for strings)
        if operator == "$contains":
            if isinstance(value, str) and isinstance(target, str):
                return target.lower() in value.lower()
            if isinstance(value, (list, tuple)):
                return target in value
            return False

        # Existence check
        if operator == "$exists":
            return bool(target)  # $exists: true matches any non-None value

        return False

    @classmethod
    def _matches_condition(cls, node: SampleNode, field: str, condition: Any) -> bool:
        """
        Check if node matches a single field condition.

        :param node: SampleNode to check
        :param field: Property name
        :param condition: Condition value or operator dict
        :return: True if matches
        """
        # Get value from node
        value = node.get_property(field)

        # Simple equality check
        if not isinstance(condition, dict):
            return value == condition

        # Operator-based checks
        for operator, target in condition.items():
            if not cls._matches_operator(value, operator, target):
                return False

        return True

    @classmethod
    def matches(cls, node: SampleNode, query: dict) -> bool:
        """
        Check if node matches filter query.

        :param node: SampleNode to check
        :param query: Filter query dict
        :return: True if node matches all conditions
        """
        # Handle logical operators
        if "$and" in query:
            return all(cls.matches(node, sub_query) for sub_query in query["$and"])

        if "$or" in query:
            return any(cls.matches(node, sub_query) for sub_query in query["$or"])

        # Check all field conditions (implicit AND)
        for field, condition in query.items():
            if field.startswith("$"):  # Skip logical operators
                continue
            if not cls._matches_condition(node, field, condition):
                return False

        return True

    @classmethod
    def filter_nodes(
        cls, nodes: Iterator[SampleNode], query: dict
    ) -> Iterator[SampleNode]:
        """
        Filter nodes matching query (generator for memory efficiency).

        :param nodes: Iterator of SampleNodes
        :param query: Filter query dict
        :return: Iterator of matching nodes
        """
        for node in nodes:
            if cls.matches(node, query):
                yield node


class SampleManager:
    """
    Main interface for node-based sample management.

    SampleManager provides a flexible, NoSQL-inspired approach to managing samples
    from multiple heterogeneous datasets. It replaces the table-based MetadataManager
    with a node-based system that handles varying experimental condition structures.

    Key Features:
    - Load samples from DataCard or DuckDB
    - Filter using MongoDB-style queries
    - Create and manage ActiveSets
    - Export to DuckDB for SQL analysis
    - Handle multi-dataset scenarios

    Example:
        >>> manager = SampleManager()
        >>> manager.load_from_datacard("BrentLab/harbison_2004", "harbison_2004")
        >>> active = manager.filter_all({"carbon_source": {"$contains": "glucose"}})
        >>> print(f"Found {len(active)} samples")
    """

    def __init__(
        self,
        duckdb_conn: duckdb.DuckDBPyConnection | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize SampleManager.

        :param duckdb_conn: Optional shared DuckDB connection for integration with HfQueryAPI
        :param cache_dir: Optional cache directory for DataCard fetches
        """
        self._collection = SampleNodeCollection()
        self._active_sets: dict[str, ActiveSet] = {}
        self._duckdb_conn = (
            duckdb_conn if duckdb_conn is not None else duckdb.connect(":memory:")
        )
        self._cache_dir = cache_dir
        self._flattener = ConditionFlattener()

    def get_active_configs(self) -> list[tuple[str, str]]:
        """
        Get list of loaded (repo_id, config_name) pairs.

        :return: List of dataset keys
        """
        return self._collection.get_dataset_keys()

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of loaded samples.

        Returns DataFrame with columns:
        - repo_id: Repository ID
        - config_name: Configuration name
        - sample_count: Number of samples
        - properties: Common properties available

        :return: Summary DataFrame
        """
        rows = []
        for repo_id, config_name in self.get_active_configs():
            count = self._collection.count_dataset_nodes(repo_id, config_name)
            # Sample first node to get property names
            first_node = next(
                self._collection.iter_dataset_nodes(repo_id, config_name), None
            )
            properties = list(first_node.properties.keys()) if first_node else []

            rows.append(
                {
                    "repo_id": repo_id,
                    "config_name": config_name,
                    "sample_count": count,
                    "properties": properties,
                }
            )

        return pd.DataFrame(rows)

    def filter_all(self, query: dict, name: str | None = None) -> ActiveSet:
        """
        Filter across all loaded samples.

        :param query: Filter query dict (MongoDB-style)
        :param name: Optional name for ActiveSet
        :return: ActiveSet with matching samples
        """
        matching_nodes = SampleFilter.filter_nodes(
            self._collection.iter_all_nodes(), query
        )
        sample_ids = {node.global_id() for node in matching_nodes}

        return ActiveSet(
            sample_ids=sample_ids,
            name=name or "filtered_samples",
            source_filter=query,
        )

    def filter_dataset(
        self,
        repo_id: str,
        config_name: str,
        query: dict,
        name: str | None = None,
    ) -> ActiveSet:
        """
        Filter samples within specific dataset.

        :param repo_id: Repository ID
        :param config_name: Config name
        :param query: Filter query dict
        :param name: Optional name for ActiveSet
        :return: ActiveSet with matching samples
        """
        matching_nodes = SampleFilter.filter_nodes(
            self._collection.iter_dataset_nodes(repo_id, config_name), query
        )
        sample_ids = {node.global_id() for node in matching_nodes}

        return ActiveSet(
            sample_ids=sample_ids,
            name=name or f"{config_name}_filtered",
            source_filter=query,
        )

    def get_sample(self, global_id: str) -> SampleNode | None:
        """
        Retrieve specific sample by global ID.

        :param global_id: Global ID in format {repo_id}:{config_name}:{sample_id}
        :return: SampleNode or None if not found
        """
        return self._collection.get_node_by_global_id(global_id)

    def get_samples_by_ids(self, sample_ids: list[str]) -> list[SampleNode]:
        """
        Batch retrieve samples.

        :param sample_ids: List of global IDs
        :return: List of SampleNodes (may be shorter if some not found)
        """
        nodes = []
        for global_id in sample_ids:
            node = self.get_sample(global_id)
            if node is not None:
                nodes.append(node)
        return nodes

    def save_active_set(self, name: str, active_set: ActiveSet):
        """
        Save named active set for later use.

        :param name: Name to save under
        :param active_set: ActiveSet to save
        """
        self._active_sets[name] = active_set

    def get_active_set(self, name: str) -> ActiveSet | None:
        """
        Retrieve saved active set.

        :param name: Name of saved set
        :return: ActiveSet or None if not found
        """
        return self._active_sets.get(name)

    def list_active_sets(self) -> list[str]:
        """
        List all saved active set names.

        :return: List of set names
        """
        return list(self._active_sets.keys())

    def get_property_distribution(
        self,
        property_name: str,
        dataset_filter: tuple[str, str] | None = None,
    ) -> dict[Any, int]:
        """
        Get value distribution for a property.

        :param property_name: Property name to analyze
        :param dataset_filter: Optional (repo_id, config_name) to limit to specific dataset
        :return: Dict mapping values to counts
        """
        distribution: dict[Any, int] = {}

        if dataset_filter:
            nodes = self._collection.iter_dataset_nodes(
                dataset_filter[0], dataset_filter[1]
            )
        else:
            nodes = self._collection.iter_all_nodes()

        for node in nodes:
            value = node.get_property(property_name, "missing")
            distribution[value] = distribution.get(value, 0) + 1

        return distribution

    def __repr__(self) -> str:
        """String representation showing loaded datasets and sample count."""
        total = self._collection.count_total_nodes()
        datasets = len(self.get_active_configs())
        return f"SampleManager({datasets} datasets, {total} samples)"
