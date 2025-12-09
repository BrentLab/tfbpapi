"""Pydantic models for dataset card validation."""

import warnings
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetType(str, Enum):
    """Supported dataset types."""

    GENOMIC_FEATURES = "genomic_features"
    ANNOTATED_FEATURES = "annotated_features"
    GENOME_MAP = "genome_map"
    METADATA = "metadata"
    QC_DATA = "qc_data"


class FieldRole(str, Enum):
    """Valid role values for feature fields."""

    REGULATOR_IDENTIFIER = "regulator_identifier"
    TARGET_IDENTIFIER = "target_identifier"
    QUANTITATIVE_MEASURE = "quantitative_measure"
    EXPERIMENTAL_CONDITION = "experimental_condition"
    GENOMIC_COORDINATE = "genomic_coordinate"


class GrowthStage(str, Enum):
    """Common growth stages."""

    MID_LOG_PHASE = "mid_log_phase"
    EARLY_LOG_PHASE = "early_log_phase"
    LATE_LOG_PHASE = "late_log_phase"
    STATIONARY_PHASE = "stationary_phase"
    EARLY_STATIONARY_PHASE = "early_stationary_phase"
    OVERNIGHT_STATIONARY_PHASE = "overnight_stationary_phase"
    MID_LOG = "mid_log"
    EARLY_LOG = "early_log"
    LATE_LOG = "late_log"
    EXPONENTIAL_PHASE = "exponential_phase"


class CompoundInfo(BaseModel):
    """Information about a chemical compound in media."""

    compound: str = Field(..., description="Chemical compound name")
    concentration_percent: float | None = Field(
        default=None, description="Concentration as percentage (w/v)"
    )
    concentration_g_per_l: float | None = Field(
        default=None, description="Concentration in grams per liter"
    )
    concentration_molar: float | None = Field(
        default=None, description="Concentration in molar (M)"
    )
    specifications: list[str] | None = Field(
        default=None,
        description="Additional specifications (e.g., 'without_amino_acids')",
    )


class MediaAdditiveInfo(BaseModel):
    """Information about media additives (e.g., butanol for filamentation)."""

    compound: str = Field(..., description="Additive compound name")
    concentration_percent: float | None = Field(
        default=None, description="Concentration as percentage (w/v)"
    )
    description: str | None = Field(
        default=None, description="Additional context about the additive"
    )


class MediaInfo(BaseModel):
    """Growth media specification."""

    name: str = Field(
        ...,
        description="Canonical or descriptive media name "
        "(minimal, synthetic_complete, YPD, etc.)",
    )
    carbon_source: list[CompoundInfo] | None = Field(
        default=None, description="Carbon source compounds and concentrations"
    )
    nitrogen_source: list[CompoundInfo] | None = Field(
        default=None, description="Nitrogen source compounds and concentrations"
    )
    phosphate_source: list[CompoundInfo] | None = Field(
        default=None, description="Phosphate source compounds and concentrations"
    )
    additives: list[MediaAdditiveInfo] | None = Field(
        default=None,
        description="Additional media components (e.g., butanol for filamentation)",
    )

    @field_validator("carbon_source", "nitrogen_source", mode="before")
    @classmethod
    def validate_compound_list(cls, v):
        """Validate compound lists and handle 'unspecified' strings."""
        if v is None:
            return None
        if isinstance(v, str):
            if v == "unspecified":
                warnings.warn(
                    "Compound source specified as string 'unspecified'. "
                    "Should be null/omitted or a structured list.",
                    UserWarning,
                )
                return None
            # Try to parse as single compound
            return [{"compound": v}]
        return v


class GrowthPhaseInfo(BaseModel):
    """Growth phase information at harvest."""

    od600: float | None = Field(
        default=None, description="Optical density at 600nm at harvest"
    )
    od600_tolerance: float | None = Field(
        default=None, description="Measurement tolerance for OD600"
    )
    stage: str | None = Field(
        default=None, description="Growth stage (preferred field name)"
    )
    phase: str | None = Field(
        default=None,
        description="Growth stage (alias for 'stage' for backward compatibility)",
    )
    description: str | None = Field(
        default=None, description="Additional context about growth phase"
    )

    @field_validator("stage", "phase", mode="before")
    @classmethod
    def validate_stage(cls, v):
        """Validate stage and warn if not a common value."""
        if v is None:
            return None
        # Get the list of known stages from the enum
        known_stages = {stage.value for stage in GrowthStage}
        if v not in known_stages:
            warnings.warn(
                f"Growth stage '{v}' not in recognized stages: {known_stages}",
                UserWarning,
            )
        return v

    @model_validator(mode="after")
    def check_stage_phase_consistency(self):
        """Ensure stage and phase are consistent if both provided."""
        if self.stage and self.phase and self.stage != self.phase:
            raise ValueError(
                "Inconsistent growth phase: "
                f"stage='{self.stage}' vs phase='{self.phase}'"
            )
        return self


class ChemicalTreatmentInfo(BaseModel):
    """Chemical treatment applied to cultures."""

    compound: str = Field(..., description="Chemical compound name")
    concentration_percent: float | None = Field(
        default=None, description="Concentration as percentage"
    )
    concentration_molar: float | None = Field(
        default=None, description="Concentration in molar (M)"
    )
    duration_minutes: int | None = Field(
        default=None, description="Duration in minutes"
    )
    duration_hours: float | None = Field(default=None, description="Duration in hours")
    target_pH: float | None = Field(
        default=None, description="Target pH for pH adjustments"
    )
    description: str | None = Field(
        default=None, description="Additional context about the treatment"
    )


class DrugTreatmentInfo(ChemicalTreatmentInfo):
    """Drug treatment - same structure as chemical treatment."""

    pass


class HeatTreatmentInfo(BaseModel):
    """Heat treatment information."""

    duration_minutes: int = Field(
        ..., description="Duration of heat treatment in minutes"
    )
    description: str | None = Field(
        default=None, description="Additional description of treatment"
    )


class TemperatureShiftInfo(BaseModel):
    """Temperature shift for heat shock experiments."""

    initial_temperature_celsius: float = Field(
        ..., description="Initial cultivation temperature in Celsius"
    )
    temperature_shift_celsius: float = Field(
        ..., description="Temperature after shift in Celsius"
    )
    temperature_shift_duration_minutes: int | None = Field(
        default=None, description="Duration of temperature shift in minutes"
    )
    description: str | None = Field(
        default=None, description="Additional context about the temperature shift"
    )


class InductionInfo(BaseModel):
    """Induction information for expression systems."""

    inducer: CompoundInfo = Field(..., description="Inducer compound and concentration")
    duration_hours: float | None = Field(
        default=None, description="Duration of induction in hours"
    )
    duration_minutes: int | None = Field(
        default=None, description="Duration of induction in minutes"
    )
    description: str | None = Field(
        default=None, description="Additional context about the induction"
    )


class EnvironmentalConditions(BaseModel):
    """Environmental conditions for sample cultivation."""

    temperature_celsius: float | None = Field(
        default=None, description="Cultivation temperature in Celsius"
    )
    cultivation_method: str | None = Field(
        default=None,
        description="Cultivation method (e.g., 'batch_culture', 'chemostat')",
    )
    growth_phase_at_harvest: GrowthPhaseInfo | None = Field(
        default=None, description="Growth phase at time of harvest"
    )
    media: MediaInfo | None = Field(
        default=None, description="Growth media specification"
    )
    chemical_treatment: ChemicalTreatmentInfo | None = Field(
        default=None, description="Chemical treatment applied"
    )
    drug_treatment: DrugTreatmentInfo | None = Field(
        default=None,
        description="Drug treatment applied (same structure as chemical_treatment)",
    )
    heat_treatment: HeatTreatmentInfo | None = Field(
        default=None, description="Heat treatment applied"
    )
    temperature_shift: TemperatureShiftInfo | None = Field(
        default=None, description="Temperature shift for heat shock experiments"
    )
    induction: InductionInfo | None = Field(
        default=None, description="Induction system for expression experiments"
    )
    incubation_duration_hours: float | None = Field(
        default=None, description="Total incubation duration in hours"
    )
    incubation_duration_minutes: int | None = Field(
        default=None, description="Total incubation duration in minutes"
    )
    description: str | None = Field(
        default=None, description="Additional descriptive information"
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def warn_extra_fields(self):
        """Warn about any extra fields not in the specification."""
        known_fields = {
            "temperature_celsius",
            "cultivation_method",
            "growth_phase_at_harvest",
            "media",
            "chemical_treatment",
            "drug_treatment",
            "heat_treatment",
            "temperature_shift",
            "induction",
            "incubation_duration_hours",
            "incubation_duration_minutes",
            "description",
        }
        extra = set(self.__dict__.keys()) - known_fields
        if extra:
            warnings.warn(
                f"EnvironmentalConditions contains non-standard fields: {extra}. "
                "Consider adding to specification.",
                UserWarning,
            )
        return self


class ExperimentalConditions(BaseModel):
    """Experimental conditions including environmental and other parameters."""

    environmental_conditions: EnvironmentalConditions | None = Field(
        default=None, description="Environmental cultivation conditions"
    )
    strain_background: str | dict[str, Any] | None = Field(
        default=None,
        description="Strain background information (string or flexible dict structure)",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def warn_extra_fields(self):
        """Warn about any extra fields not in the specification."""
        known_fields = {"environmental_conditions", "strain_background"}
        extra = set(self.__dict__.keys()) - known_fields
        if extra:
            warnings.warn(
                f"ExperimentalConditions contains non-standard fields: {extra}. "
                "Consider adding to specification.",
                UserWarning,
            )
        return self


class ClassLabelType(BaseModel):
    """Categorical data type with class labels."""

    names: list[str] = Field(..., description="List of possible class names")


class FeatureInfo(BaseModel):
    """Information about a dataset feature/column."""

    name: str = Field(..., description="Column name in the data")
    dtype: str | dict[str, ClassLabelType] = Field(
        ...,
        description="Data type (string, int64, float64, etc.) "
        "or categorical class labels",
    )
    description: str = Field(..., description="Detailed description of the field")
    role: FieldRole | None = Field(
        default=None,
        description="Semantic role of the feature (e.g., 'target_identifier', "
        "'regulator_identifier', 'quantitative_measure')",
    )
    definitions: dict[str, Any] | None = Field(
        default=None,
        description="Definitions for categorical field values "
        "with experimental conditions",
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
                        "Invalid class_label structure: expected dict "
                        f"with 'names' key, got {class_label_data}"
                    )
            else:
                raise ValueError(
                    "Unknown dtype structure: expected 'class_label' key "
                    f"in dict, got keys: {list(v.keys())}"
                )
        else:
            raise ValueError(
                "dtype must be a string or dict with "
                f"class_label info, got {type(v)}: {v}"
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
    experimental_conditions: ExperimentalConditions | None = Field(
        default=None, description="Experimental conditions for this config"
    )
    data_files: list[DataFileInfo] = Field(..., description="Data file information")
    dataset_info: DatasetInfo = Field(..., description="Dataset structure information")

    @field_validator("applies_to")
    @classmethod
    def applies_to_only_for_metadata(cls, v, info):
        """Validate that applies_to is only used for metadata or qc_data configs."""
        if v is not None:
            dataset_type = info.data.get("dataset_type")
            if dataset_type not in (DatasetType.METADATA, DatasetType.QC_DATA):
                raise ValueError(
                    "applies_to field is only valid "
                    "for metadata and qc_data dataset types"
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
    experimental_conditions: ExperimentalConditions | None = Field(
        default=None, description="Top-level experimental conditions for all configs"
    )
    license: str | None = Field(default=None, description="Dataset license")
    language: list[str] | None = Field(default=None, description="Dataset languages")
    tags: list[str] | None = Field(default=None, description="Descriptive tags")
    pretty_name: str | None = Field(
        default=None, description="Human-readable dataset name"
    )
    size_categories: list[str] | None = Field(
        default=None, description="Dataset size categories"
    )
    strain_information: dict[str, Any] | None = Field(
        default=None, description="Strain background information"
    )

    model_config = ConfigDict(extra="allow")

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
