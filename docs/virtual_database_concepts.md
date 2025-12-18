# Virtual Database

VirtualDB provides a unified query interface across heterogeneous datasets with
different experimental condition structures and terminologies. Each dataset
defines experimental conditions in its own way, with properties stored at
different hierarchy levels (repository, dataset, or field) and using different
naming conventions. VirtualDB uses an external YAML configuration to map these
varying structures to a common schema, normalize factor level names (e.g.,
"D-glucose", "dextrose", "glu" all become "glucose"), and enable cross-dataset
queries with standardized field names and values.

## Configuration Structure

A configuration file defines the virtual database schema with four sections:

```yaml
# ===== Repository Configurations =====
repositories:
  # Each repository defines a "table" in the virtual database
  BrentLab/harbison_2004:
    # Repository-wide properties (apply to all datasets in this repository)
    nitrogen_source:
      path: media.nitrogen_source.name

    dataset:
      # Each dataset gets its own view with standardized fields
      harbison_2004:
        # Dataset-specific properties (constant for all samples)
        phosphate_source:
          path: media.phosphate_source.compound

        # Field-level properties (vary per sample)
        carbon_source:
          field: condition
          path: media.carbon_source.compound

        # Field without path (column alias with normalization)
        environmental_condition:
          field: condition

  BrentLab/kemmeren_2014:
    dataset:
      kemmeren_2014:
        # Same logical fields, different physical paths
        carbon_source:
          path: media.carbon_source.compound
        temperature_celsius:
          path: temperature_celsius

# ===== Normalization Rules =====
# Map varying terminologies to standardized values
factor_aliases:
  carbon_source:
    glucose: [D-glucose, glu, dextrose]
    galactose: [D-galactose, gal]

# Handle missing values with defaults
missing_value_labels:
  carbon_source: "unspecified"

# ===== Documentation =====
description:
  carbon_source: The carbon source provided to the cells during growth
```

### Property Hierarchy

Properties are extracted at three hierarchy levels:

1. **Repository-wide**: Common to all datasets in a repository
   - Paths relative to repository-level `experimental_conditions`
   - Example: `path: media.nitrogen_source.name`

2. **Dataset-specific**: Specific to one dataset configuration
   - Paths relative to config-level `experimental_conditions`
   - Example: `path: media.phosphate_source.compound`

3. **Field-level**: Vary per sample, defined in field definitions
   - `field` specifies which field to extract from
   - `path` relative to field definitions (not `experimental_conditions`)
   - Example: `field: condition, path: media.carbon_source.compound`

**Special case**: Field without path creates a column alias
- `field: condition` (no path) to renames `condition` column, enables normalization

### Path Resolution

Paths use dot notation to navigate nested structures:

**Repository/Dataset-level** (automatically prepends `experimental_conditions.`):
- `path: temperature_celsius` to `experimental_conditions.temperature_celsius`
- `path: media.carbon_source.compound` to
  `experimental_conditions.media.carbon_source.compound`

**Field-level** (paths relative to field definitions):
- `field: condition, path: media.carbon_source.compound` to looks in field
`condition`'s definitions to navigates to `media.carbon_source.compound`

## VirtualDB Structure

VirtualDB maintains a collection of dataset-specific metadata tables, one per
configured dataset. Each table has the same structure (standardized schema) but
contains data specific to that dataset.

### Internal Structure

When materialized (or conceptually if not materialized), VirtualDB contains:

```python
{
    ("BrentLab/harbison_2004", "harbison_2004"): DataFrame(
        # Columns: sample_id, carbon_source, temperature_celsius, nitrogen_source, ...
        # Values: Normalized according to factor_aliases
        # Example rows:
        #   sample_id       carbon_source  temperature_celsius  nitrogen_source
        #   harbison_001    glucose        30                   yeast nitrogen base
        #   harbison_002    galactose      30                   yeast nitrogen base
    ),

    ("BrentLab/kemmeren_2014", "kemmeren_2014"): DataFrame(
        # Columns: sample_id, carbon_source, temperature_celsius, ...
        # Note: Different physical source paths, same logical schema
        # Example rows:
        #   sample_id       carbon_source  temperature_celsius
        #   kemmeren_001    glucose        30
        #   kemmeren_002    raffinose      30
    )
}
```

### View Materialization

By default, VirtualDB computes views on-demand. For performance, views can be
materialized (cached):

```python
# Cache all views for faster subsequent queries
vdb.materialize_views()

# Cache specific datasets
vdb.materialize([("BrentLab/harbison_2004", "harbison_2004")])

# Invalidate cache (e.g., after data updates)
vdb.invalidate_cache()
vdb.invalidate_cache([("BrentLab/harbison_2004", "harbison_2004")])
```

Materialized views are stored locally and reused for queries until invalidated.

## VirtualDB Interface

### Schema Discovery

**List all queryable fields**:
```python
from tfbpapi.virtual_db import VirtualDB

vdb = VirtualDB("config.yaml")

# All fields defined in any dataset
fields = vdb.get_fields()
# ["carbon_source", "temperature_celsius", "nitrogen_source", "phosphate_source", ...]

# Fields present in ALL datasets (common fields)
common = vdb.get_common_fields()
# ["carbon_source", "temperature_celsius"]

# Fields for specific dataset
dataset_fields = vdb.get_fields("BrentLab/harbison_2004", "harbison_2004")
# ["carbon_source", "temperature_celsius", "nitrogen_source", "phosphate_source"]
```

**Discover valid values for fields**:
```python
# Unique values across all datasets (normalized)
values = vdb.get_unique_values("carbon_source")
# ["glucose", "galactose", "raffinose", "unspecified"]

# Values broken down by dataset
values_by_dataset = vdb.get_unique_values("carbon_source", by_dataset=True)
# {
#     "BrentLab/harbison_2004": ["glucose", "galactose"],
#     "BrentLab/kemmeren_2014": ["glucose", "raffinose"]
# }
```

### Querying Data

The `query()` method is the primary interface for retrieving data from VirtualDB.

**Basic usage** (sample-level, all fields):
```python
# Query across all configured datasets
# Returns one row per sample with all configured fields
df = vdb.query(filters={"carbon_source": "glucose"})
# DataFrame: sample_id, carbon_source, temperature_celsius, nitrogen_source, ...
```

**Query specific datasets**:
```python
# Limit query to specific datasets
df = vdb.query(
    filters={"carbon_source": "glucose", "temperature_celsius": 30},
    datasets=[("BrentLab/harbison_2004", "harbison_2004")]
)
```

**Select specific fields**:
```python
# Return only specified fields
df = vdb.query(
    filters={"carbon_source": "glucose"},
    fields=["sample_id", "carbon_source", "temperature_celsius"]
)
# DataFrame: sample_id, carbon_source, temperature_celsius
```

**Complete data** (measurement-level):
```python
# Set complete=True to get all measurements, not just sample-level
# Returns many rows per sample (one per target/feature/coordinate)
df = vdb.query(
    filters={"carbon_source": "glucose"},
    complete=True
)
# DataFrame: sample_id, target, value, carbon_source, temperature_celsius, ...
# For annotated_features: target-level data for all matching samples
# For genome_map: coordinate-level data for all matching samples

# Can combine with field selection
df = vdb.query(
    filters={"carbon_source": "glucose"},
    fields=["sample_id", "target", "effect"],
    complete=True
)
# DataFrame: sample_id, target, effect
```

### Factor Alias Expansion

When querying with aliased values, VirtualDB automatically expands to all
original values:

```python
# User queries for normalized value
df = vdb.query(filters={"carbon_source": "galactose"})

# Internally expands to all aliases
# WHERE carbon_source IN ('D-galactose', 'gal', 'galactose')
```

This ensures all samples are retrieved regardless of original terminology.

### Numeric Field Filtering

Numeric fields support exact matching and range queries:

```python
# Exact match
df = vdb.query(filters={"temperature_celsius": 30})

# Range queries
df = vdb.query(filters={"temperature_celsius": (">=", 28)})
df = vdb.query(filters={"temperature_celsius": ("between", 28, 32)})

# Missing value labels (from missing_value_labels config)
df = vdb.query(filters={"temperature_celsius": "room"})
# Matches samples where temperature is None/missing
```

## Design Principles

### Virtual Views by Default

Views are computed on-demand unless explicitly materialized, but optionally cached,
This provides:

- Reduces setup time when a new instance instantiates MetadataBuilder/VirtualDB
- No storage overhead for casual queries

### External Configuration as Schema

The YAML configuration serves as a **database schema definition**:

- Defines what fields exist (logical schema)
- Maps to physical data structures (via paths)
- Specifies normalization rules (via aliases)
- Documents field semantics (via descriptions)

This separation enables:

- Schema updates without code changes

See:
- [DataCard Documentation](huggingface_datacard.md)
