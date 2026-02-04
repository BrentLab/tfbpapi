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

This is a basic example of a VirtualDB configuration YAML file:

```yaml
repositories:
  # Each repository defines a "table" in the virtual database
  BrentLab/harbison_2004:
    # REQUIRED: Specify which field is the sample identifier. At this level, it means
    # that all datasets have a field `sample_id` that uniquely identifies samples.
    sample_id:
      field: sample_id
    # Repository-wide properties (apply to all datasets in this repository)
    # Paths are explicit from the datacard root
    nitrogen_source:
      path: experimental_conditions.media.nitrogen_source.name

    dataset:
      # Each dataset gets its own view with standardized fields
      harbison_2004:
        # Dataset-specific properties (constant for all samples)
        # Explicit path from datacard/config root
        phosphate_source:
          path: experimental_conditions.media.phosphate_source.compound

        # Field-level properties (vary per sample)
        # Path is relative to field's definitions dict
        carbon_source:
          field: condition
          path: media.carbon_source.compound
          dtype: string  # Optional: specify data type

        # Field without path (column alias with normalization)
        environmental_condition:
          field: condition

        # if there is a `comparative_analysis` dataset that you want to link to
        # a given dataset, you can declare it at the dataset level
        # For more information on this section, see the section
        # 'Comparative Datasets in VirtualDB'
        comparative_analyses:
          # specify the comparative analysis repo
          - repo: BrentLab/yeast_comparative_analysis
            # and dataset
            dataset: dto
            # and the field in the comparative analysis that links back to this
            # dataset. Note that this field should have role `source_sample`, and it
            # should therefore be formatted as `repo_id;config_name;sample_id` where the
            # sample_id is derived from the field in this dataset that is specified
            # for this dataset in the `sample_id` field above.
            via_field: perturbation_id

  BrentLab/kemmeren_2014:
    dataset:
      kemmeren_2014:
        # REQUIRED: If `sample_id` isn't defined at the repo level, then it must be
        # defined at the dataset level for each dataset in the repo
        sample_id:
          field: sample_id
        # Same logical fields, different physical paths
        # Explicit path from datacard/config root
        carbon_source:
          path: experimental_conditions.media.carbon_source.compound
          dtype: string
        temperature_celsius:
          path: experimental_conditions.temperature_celsius
          dtype: numeric  # Enables numeric filtering with comparison operators

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
   - Paths relative to datacard/config root (explicit)
   - Example: `path: experimental_conditions.media.nitrogen_source.name`

2. **Dataset-specific**: Specific to one dataset configuration
   - Paths relative to datacard/config root (explicit)
   - Example: `path: experimental_conditions.media.phosphate_source.compound`

3. **Field-level**: Vary per sample, defined in field definitions
   - `field` specifies which field to extract from
   - `path` relative to that field's definitions dict
   - Example: `field: condition, path: media.carbon_source.compound`

**Special case**: Field without path creates a column alias
- `field: condition` (no path) renames `condition` column, enables normalization

### Path Resolution

Paths use dot notation to navigate nested structures:

**Repository/Dataset-level** (explicit paths from datacard root):
- `path: experimental_conditions.temperature_celsius` - access experimental conditions
- `path: experimental_conditions.media.carbon_source.compound` - nested condition data
- `path: description` - access fields outside experimental_conditions

**Field-level** (paths relative to field definitions):
- `field: condition, path: media.carbon_source.compound` looks in field
  `condition`'s definitions and navigates to `media.carbon_source.compound`

### Data Type Specifications

Field mappings support an optional `dtype` parameter to ensure proper type handling
during metadata extraction and query filtering.

**Supported dtypes**:
- `string` - Text data (default if not specified)
- `numeric` - Numeric values (integers or floating-point numbers)
- `bool` - Boolean values (true/false)

**When to use dtype**:

1. **Numeric filtering**: Required for fields used with comparison operators
   (`<`, `>`, `<=`, `>=`, `between`)
2. **Type consistency**: When source data might be extracted with incorrect type
3. **Performance**: Helps with query optimization and prevents type mismatches

**Type conversion process**:

Type conversion happens during metadata extraction:
1. Extract value from source using path
2. Convert to specified dtype if provided
3. Store in metadata DataFrame with correct type

**Example - The problem**:
```python
# Without dtype: temperature extracted as string "30"
# Comparison fails or produces incorrect results
df = vdb.query(filters={"temperature_celsius": (">", 25)})
# String comparison: "30" > 25 evaluates incorrectly
```

**Example - The solution**:
```yaml
temperature_celsius:
  path: temperature_celsius
  dtype: numeric  # Ensures numeric type for proper comparison
```

```python
# With dtype: temperature extracted as numeric 30.0
# Comparison works correctly
df = vdb.query(filters={"temperature_celsius": (">", 25)})
# Numeric comparison: 30.0 > 25 is True (correct!)
```

**Usage examples**:
```yaml
repositories:
  BrentLab/example:
    dataset:
      example_dataset:
        # String field for categorical data (explicit path from datacard root)
        strain_background:
          path: experimental_conditions.strain_background
          dtype: string

        # Numeric field for quantitative filtering
        temperature_celsius:
          path: experimental_conditions.temperature_celsius
          dtype: numeric

        # Numeric field for concentration measurements (nested path)
        drug_concentration_um:
          path: experimental_conditions.drug_treatment.concentration_um
          dtype: numeric

        # Boolean field
        is_heat_shock:
          path: experimental_conditions.is_heat_shock
          dtype: bool

        # Access fields outside experimental_conditions
        dataset_description:
          path: description
          dtype: string
```

## VirtualDB Structure

VirtualDB maintains a collection of dataset-specific metadata tables, one per
configured dataset. Each table has the same structure (standardized schema) but
contains data specific to that dataset.  

Unless directed, these tables are not stored on desk and instead generated via
query against the source parquet files. Think of them as a typical database view.

### Internal Structure

```python
{
    # Primary datasets with sample_id
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
    ),

    # Comparative datasets with parsed composite identifiers
    ("BrentLab/yeast_comparative_analysis", "dto"): DataFrame(
        # Original composite ID columns preserved
        # Columns: binding_id, perturbation_id, dto_fdr, dto_empirical_pvalue, ...
        # Example rows:
        #   binding_id                                           perturbation_id                               dto_fdr
        #   BrentLab/harbison_2004;harbison_2004;harbison_001   BrentLab/kemmeren_2014;kemmeren_2014;sample_42  0.001
        #   BrentLab/harbison_2004;harbison_2004;harbison_002   BrentLab/kemmeren_2014;kemmeren_2014;sample_43  0.045
        #
        # When materialized with foreign keys, additional parsed columns are created:
        # Columns: binding_id, binding_repo_id, binding_config_name, binding_sample_id,
        #          perturbation_id, perturbation_repo_id, perturbation_config_name, perturbation_sample_id,
        #          dto_fdr, dto_empirical_pvalue, ...
        # Example rows:
        #   binding_repo_id         binding_config_name  binding_sample_id  dto_fdr
        #   BrentLab/harbison_2004  harbison_2004        harbison_001       0.001
        #   BrentLab/harbison_2004  harbison_2004        harbison_002       0.045
    )
}
```

### View Materialization

Tables can be cached for faster subsequent queries via materialization:

```python
# Cache all views for faster subsequent queries
vdb.materialize_views()

# Cache specific datasets
vdb.materialize([("BrentLab/harbison_2004", "harbison_2004")])

# Invalidate cache (e.g., after data updates)
vdb.invalidate_cache()
vdb.invalidate_cache([("BrentLab/harbison_2004", "harbison_2004")])
```

Materialized views are stored locally and reused for queries.

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
original values specified in the configuration:

```python
# User queries for normalized value
df = vdb.query(filters={"carbon_source": "galactose"})

# Internally expands to all aliases
# WHERE carbon_source IN ('D-galactose', 'gal', 'galactose')
```

### Numeric Field Filtering

Numeric fields support exact matching and range queries:

```python
# Exact match
df = vdb.query(filters={"temperature_celsius": 30})

# Range queries
df = vdb.query(filters={"temperature_celsius": (">=", 28)})
# inclusive of the boundaries, ie [28, 32]
df = vdb.query(filters={"temperature_celsius": ("between", 28, 32)})

# Missing value labels. This analogous to how factor_aliases work. In this case, it
# will return where the temprature_celsius is missing/None/Null/NaN/etc and/or the
# value matches the specified label, in this case "room". If the missing value label
# is a character value and the field is a numeric field, then only missing values will
# be matched.
df = vdb.query(filters={"temperature_celsius": "room"})
# Matches samples where temperature is None/missing
```

## Comparative Datasets in VirtualDB

Comparative datasets differ from other dataset types in that they represent
relationships between samples across datasets rather than individual samples.
Each row relates 2+ samples from other datasets.

### Structure

Comparative datasets use `source_sample` fields instead of a single `sample_id`:
- Multiple fields with `role: source_sample`
- Each contains composite identifier: `"repo_id;config_name;sample_id"`
- Example: `binding_id = "BrentLab/harbison_2004;harbison_2004;42"`

### Querying Comparative Data

Comparative datasets can be queried in two ways: **direct queries** for analysis
results, and **field-based queries** to enrich primary dataset queries with
comparative metrics.

#### Direct Queries

Query the comparative dataset directly to find analysis results:

```python
# Find significant DTO results across all experiments
dto_results = vdb.query(
    datasets=[("BrentLab/yeast_comparative_analysis", "dto")],
    filters={"dto_fdr": ("<", 0.05)},
    complete=True
)
# Returns: binding_id, perturbation_id, dto_fdr, dto_empirical_pvalue,
#          binding_rank_threshold, perturbation_rank_threshold, ...

# Filter by source dataset
dto_for_harbison = vdb.query(
    datasets=[("BrentLab/yeast_comparative_analysis", "dto")],
    filters={"binding_id": ("contains", "harbison_2004")},
    complete=True
)

# Combine filters on both metrics and source samples
high_quality_dto = vdb.query(
    datasets=[("BrentLab/yeast_comparative_analysis", "dto")],
    filters={
        "dto_fdr": ("<", 0.01),
        "binding_id": ("contains", "callingcards")
    },
    complete=True
)
```

#### Field-based Queries

```python
# Query binding data, automatically include DTO metrics
binding_with_dto = vdb.query(
    datasets=[("BrentLab/callingcards", "annotated_features")],
    filters={"regulator_locus_tag": "YJR060W"},
    fields=["sample_id", "target_locus_tag", "binding_score", "dto_fdr"],
    complete=True
)
# Returns binding data WITH dto_fdr joined automatically via composite ID

# Query perturbation data, include derived significance field
perturbation_with_significance = vdb.query(
    datasets=[("BrentLab/hackett_2020", "hackett_2020")],
    filters={"regulator_locus_tag": "YJR060W"},
    fields=["sample_id", "target_locus_tag", "log2fc", "is_significant"],
    complete=True
)
# Returns perturbation data WITH is_significant (computed from dto_fdr < 0.05)
```

### Configuration

Comparative datasets work differently - 
**primary datasets declare which comparative datasets reference them**:

```yaml
repositories:
  # Primary dataset (e.g., binding data)
  BrentLab/callingcards:
    dataset:
      annotated_features:
        # REQUIRED: Specify which field is the sample identifier
        sample_id:
          field: sample_id

        # OPTIONAL: Declare comparative analyses that include this dataset
        comparative_analyses:
          - repo: BrentLab/yeast_comparative_analysis
            dataset: dto
            via_field: binding_id
            # VirtualDB knows composite format: "BrentLab/callingcards;annotated_features;<sample_id>"

        # Regular fields
        regulator_locus_tag:
          field: regulator_locus_tag
        # ... other fields

  # Another primary dataset (e.g., perturbation data)
  BrentLab/hu_2007_reimand_2010:
    dataset:
      data:
        sample_id:
          field: sample_id

        comparative_analyses:
          - repo: BrentLab/yeast_comparative_analysis
            dataset: dto
            via_field: perturbation_id

        # Regular fields
        # ... other fields

  # Comparative dataset - OPTIONAL field mappings for renaming/aliasing
  BrentLab/yeast_comparative_analysis:
    dataset:
      dto:
        # Optional: Rename fields for clarity or add derived columns
        fdr:
          field: dto_fdr  # Rename dto_fdr to fdr

        empirical_pvalue:
          field: dto_empirical_pvalue  # Rename for consistency

        is_significant:
          # Derived field: computed from dto_fdr
          expression: "dto_fdr < 0.05"
```

## See Also
- [DataCard Documentation](huggingface_datacard.md)
