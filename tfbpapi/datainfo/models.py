"""Pydantic models for dataset card validation."""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetType(str, Enum):
    """Supported dataset types."""

    GENOMIC_FEATURES = "genomic_features"
    ANNOTATED_FEATURES = "annotated_features"
    GENOME_MAP = "genome_map"
    METADATA = "metadata"


class ClassLabelType(BaseModel):
    """Categorical data type with class labels."""

    names: list[str] = Field(..., description="List of possible class names")


class FeatureInfo(BaseModel):
    """Information about a dataset feature/column."""

    name: str = Field(..., description="Column name in the data")
    dtype: str | dict[str, ClassLabelType] = Field(
        ...,
        description="Data type (string, int64, float64, etc.) or categorical class labels",
    )
    description: str = Field(..., description="Detailed description of the field")
    role: str | None = Field(
        default=None,
        description="Semantic role of the feature (e.g., 'target_identifier', 'regulator_identifier', 'quantitative_measure')",
    )

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v):
        """Validate and normalize dtype field."""
        if isinstance(v, str):
            return v
        elif isinstance(v, dict):
            # Handle class_label structure
            if "class_label" in v:
                # Convert to our ClassLabelType structure
                class_label_data = v["class_label"]
                if isinstance(class_label_data, dict) and "names" in class_label_data:
                    return {"class_label": ClassLabelType(**class_label_data)}
                else:
                    raise ValueError(
                        f"Invalid class_label structure: expected dict with 'names' key, got {class_label_data}"
                    )
            else:
                raise ValueError(
                    f"Unknown dtype structure: expected 'class_label' key in dict, got keys: {list(v.keys())}"
                )
        else:
            raise ValueError(
                f"dtype must be a string or dict with class_label info, got {type(v)}: {v}"
            )

    def get_dtype_summary(self) -> str:
        """Get a human-readable summary of the data type."""
        if isinstance(self.dtype, str):
            return self.dtype
        elif isinstance(self.dtype, dict) and "class_label" in self.dtype:
            names = self.dtype["class_label"].names
            return f"categorical ({len(names)} classes: {', '.join(names)})"
        else:
            return str(self.dtype)


class PartitioningInfo(BaseModel):
    """Partitioning configuration for datasets."""

    enabled: bool = Field(default=False, description="Whether partitioning is enabled")
    partition_by: list[str] | None = Field(
        default=None, description="Partition column names"
    )
    path_template: str | None = Field(
        default=None, description="Path template for partitioned files"
    )


class DataFileInfo(BaseModel):
    """Information about data files."""

    split: str = Field(default="train", description="Dataset split name")
    path: str = Field(..., description="Path to data file(s)")


class DatasetInfo(BaseModel):
    """Dataset structure information."""

    features: list[FeatureInfo] = Field(..., description="Feature definitions")
    partitioning: PartitioningInfo | None = Field(
        default=None, description="Partitioning configuration"
    )


class DatasetConfig(BaseModel):
    """Configuration for a dataset within a repository."""

    config_name: str = Field(..., description="Unique configuration identifier")
    description: str = Field(..., description="Human-readable description")
    dataset_type: DatasetType = Field(..., description="Type of dataset")
    default: bool = Field(
        default=False, description="Whether this is the default config"
    )
    applies_to: list[str] | None = Field(
        default=None, description="Configs this metadata applies to"
    )
    metadata_fields: list[str] | None = Field(
        default=None, description="Fields for embedded metadata extraction"
    )
    data_files: list[DataFileInfo] = Field(..., description="Data file information")
    dataset_info: DatasetInfo = Field(..., description="Dataset structure information")

    @field_validator("applies_to")
    @classmethod
    def applies_to_only_for_metadata(cls, v, info):
        """Validate that applies_to is only used for metadata configs."""
        if v is not None:
            dataset_type = info.data.get("dataset_type")
            if dataset_type != DatasetType.METADATA:
                raise ValueError(
                    "applies_to field is only valid for metadata dataset types"
                )
        return v

    @field_validator("metadata_fields")
    @classmethod
    def metadata_fields_validation(cls, v):
        """Validate metadata_fields usage."""
        if v is not None and len(v) == 0:
            raise ValueError("metadata_fields cannot be empty list, use None instead")
        return v


class BasicMetadata(BaseModel):
    """Basic dataset metadata."""

    license: str | None = Field(default=None, description="Dataset license")
    language: list[str] | None = Field(default=None, description="Dataset languages")
    tags: list[str] | None = Field(default=None, description="Descriptive tags")
    pretty_name: str | None = Field(
        default=None, description="Human-readable dataset name"
    )
    size_categories: list[str] | None = Field(
        default=None, description="Dataset size categories"
    )


class DatasetCard(BaseModel):
    """Complete dataset card model."""

    configs: list[DatasetConfig] = Field(..., description="Dataset configurations")
    license: str | None = Field(default=None, description="Dataset license")
    language: list[str] | None = Field(default=None, description="Dataset languages")
    tags: list[str] | None = Field(default=None, description="Descriptive tags")
    pretty_name: str | None = Field(
        default=None, description="Human-readable dataset name"
    )
    size_categories: list[str] | None = Field(
        default=None, description="Dataset size categories"
    )

    @field_validator("configs")
    @classmethod
    def configs_not_empty(cls, v):
        """Ensure at least one config is present."""
        if not v:
            raise ValueError("At least one dataset configuration is required")
        return v

    @field_validator("configs")
    @classmethod
    def unique_config_names(cls, v):
        """Ensure config names are unique."""
        names = [config.config_name for config in v]
        if len(names) != len(set(names)):
            raise ValueError("Configuration names must be unique")
        return v

    @field_validator("configs")
    @classmethod
    def at_most_one_default(cls, v):
        """Ensure at most one config is marked as default."""
        defaults = [config for config in v if config.default]
        if len(defaults) > 1:
            raise ValueError("At most one configuration can be marked as default")
        return v

    def get_config_by_name(self, name: str) -> DatasetConfig | None:
        """Get a configuration by name."""
        for config in self.configs:
            if config.config_name == name:
                return config
        return None

    def get_configs_by_type(self, dataset_type: DatasetType) -> list[DatasetConfig]:
        """Get all configurations of a specific type."""
        return [
            config for config in self.configs if config.dataset_type == dataset_type
        ]

    def get_default_config(self) -> DatasetConfig | None:
        """Get the default configuration if one exists."""
        defaults = [config for config in self.configs if config.default]
        return defaults[0] if defaults else None

    def get_data_configs(self) -> list[DatasetConfig]:
        """Get all non-metadata configurations."""
        return [
            config
            for config in self.configs
            if config.dataset_type != DatasetType.METADATA
        ]

    def get_metadata_configs(self) -> list[DatasetConfig]:
        """Get all metadata configurations."""
        return [
            config
            for config in self.configs
            if config.dataset_type == DatasetType.METADATA
        ]


class ExtractedMetadata(BaseModel):
    """Metadata extracted from datasets."""

    config_name: str = Field(..., description="Source configuration name")
    field_name: str = Field(
        ..., description="Field name the metadata was extracted from"
    )
    values: set[str] = Field(..., description="Unique values found")
    extraction_method: str = Field(..., description="How the metadata was extracted")

    model_config = ConfigDict(
        # Allow sets in JSON serialization
        json_encoders={set: list}
    )


class MetadataRelationship(BaseModel):
    """Relationship between a data config and its metadata."""

    data_config: str = Field(..., description="Data configuration name")
    metadata_config: str = Field(..., description="Metadata configuration name")
    relationship_type: str = Field(
        ..., description="Type of relationship (explicit, embedded)"
    )
