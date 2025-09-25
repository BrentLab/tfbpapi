# HuggingFace Dataset Card Format

This document describes the expected YAML metadata format for HuggingFace dataset repositories used with the tfbpapi package. The metadata is defined in the repository's README.md file frontmatter and provides structured information about the dataset configuration and contents.

## Required Top-Level Fields

### Basic Metadata
```yaml
license: mit                    # Dataset license
language:                      # Languages (usually 'en' for scientific data)
- en
tags:                          # Descriptive tags for discoverability
- biology
- genomics
- yeast
- transcription-factors
pretty_name: "Dataset Name"    # Human-readable dataset name
size_categories:               # Dataset size category
- 100K<n<1M
```

### Dataset Configurations

The most important section is the `configs` array, which defines multiple dataset configurations within a single repository:

```yaml
configs:
- config_name: config_identifier
  description: Human-readable description
  dataset_type: [genomic_features|annotated_features|genome_map|metadata]
  default: true                # Optional: marks default config
  applies_to: [config1, config2]  # Optional: for metadata configs only
  metadata_fields: [field1, field2]  # Optional: for embedded metadata extraction
  data_files:
  - split: train               # Usually 'train' for scientific data
    path: file.parquet         # Path to data file(s)
  dataset_info:
    features: [...]            # Feature definitions (see below)
    partitioning: [...]        # Optional: for partitioned datasets
```

## Dataset Types

The tfbpapi recognizes four main dataset types via the `dataset_type` field:

### 1. `genomic_features`
Static information about genomic features (genes, promoters, etc.)
- **Use case**: Gene annotations, regulatory classifications, static feature data
- **Structure**: One row per genomic feature
- **Required fields**: Usually includes gene identifiers, coordinates, classifications

### 2. `annotated_features`
Quantitative data associated with genomic features
- **Use case**: Expression data, binding scores, differential expression results
- **Structure**: Regulator-target pairs or feature-condition measurements
- **Common fields**: `regulator_*`, `target_*`, quantitative measurements

### 3. `genome_map`
Position-level data across genomic coordinates
- **Use case**: Signal tracks, coverage data, genome-wide binding profiles
- **Structure**: Position-value pairs, often large datasets
- **Required fields**: `chr` (chromosome), `pos` (position), signal values

### 4. `metadata`
Experimental metadata and sample descriptions
- **Use case**: Sample information, experimental conditions, protocol details
- **Structure**: One row per experiment/sample
- **Common fields**: Sample identifiers, experimental conditions, publication info
- **Special field**: `applies_to` - Optional list of config names this metadata applies to

## Feature Definitions

Each config must include detailed feature definitions in `dataset_info.features`:

```yaml
dataset_info:
  features:
  - name: field_name           # Column name in the data
    dtype: string              # Data type (string, int64, float64, etc.)
    description: "Detailed description of what this field contains"
    role: "target_identifier"  # Optional: semantic role of the feature
```

### Common Data Types
- `string`: Text data, identifiers, categories
- `int64`: Integer values
- `float64`: Decimal numbers, measurements
- `int32`, `float32`: For large datasets where precision/memory matters

### Naming Conventions

**Gene/Feature Identifiers:**
- `*_locus_tag`: Systematic gene identifiers (e.g., "YDL227C")
- `*_symbol`: Standard gene symbols (e.g., "HO")
- `*_id`: Alternative identifier systems

**Regulator-Target Relationships:**
- `regulator_*`: Fields describing the regulatory factor
- `target_*`: Fields describing the target gene/feature

**Genomic Coordinates:**
- `chr`: Chromosome identifier
- `start`, `end`: Genomic coordinates
- `pos`: Single position
- `strand`: Strand information (+ or -)

### Feature Roles

The optional `role` field provides semantic meaning to features, especially useful for `annotated_features` datasets:

**Standard Roles:**
- `target_identifier`: Identifies target genes/features (e.g., target_locus_tag, target_symbol)
- `regulator_identifier`: Identifies regulatory factors (e.g., regulator_locus_tag, regulator_symbol)
- `quantitative_measure`: Quantitative measurements (e.g., binding_score, expression_level, p_value)
- `experimental_condition`: Experimental conditions or metadata
- `genomic_coordinate`: Positional information (chr, start, end, pos)

## Partitioned Datasets

For large datasets (eg most genome_map datasets), use partitioning:

```yaml
dataset_info:
  partitioning:
    enabled: true
    partition_by: ["accession"]  # Partition column(s)
    path_template: "data/accession={accession}/*.parquet"
```

This allows efficient querying of subsets without loading the entire dataset.

## Metadata 

### Metadata Relationships with `applies_to`

For metadata configs, you can explicitly specify which other configs the metadata applies to using the `applies_to` field. This provides more control than automatic type-based matching.

```yaml
configs:
# Data configs
- config_name: genome_map_data
  dataset_type: genome_map
  # ... rest of config

- config_name: binding_scores
  dataset_type: annotated_features
  # ... rest of config

- config_name: expression_data
  dataset_type: annotated_features
  # ... rest of config

# Metadata config that applies to multiple data configs
- config_name: repo_metadata
  dataset_type: metadata
  applies_to: ["genome_map_data", "binding_scores", "expression_data"]
  # ... rest of config
```

### Embedded Metadata with `metadata_fields`

When no explicit metadata config exists, you can extract metadata directly from the dataset's own files using the `metadata_fields` field. This specifies which fields should be treated as metadata.

### Single File Embedded Metadata

For single parquet files, the system extracts distinct values using `SELECT DISTINCT`:

```yaml
- config_name: binding_data
  dataset_type: annotated_features
  metadata_fields: ["regulator_symbol", "experimental_condition"]
  data_files:
  - split: train
    path: binding_measurements.parquet
  dataset_info:
    features:
    - name: regulator_symbol
      dtype: string
      description: Transcription factor name
    - name: experimental_condition
      dtype: string
      description: Experimental treatment
    - name: binding_score
      dtype: float64
      description: Quantitative measurement
```

### Partitioned Dataset Embedded Metadata

For partitioned datasets, partition values are extracted from directory structure:

```yaml
- config_name: genome_map_data
  dataset_type: genome_map
  metadata_fields: ["run_accession", "regulator_symbol"]
  data_files:
  - split: train
    path: genome_map/accession=*/regulator=*/*.parquet
  dataset_info:
    features:
    - name: chr
      dtype: string
      description: Chromosome
    - name: pos
      dtype: int32
      description: Position
    - name: signal
      dtype: float32
      description: Signal intensity
    partitioning:
      enabled: true
      partition_by: ["run_accession", "regulator_symbol"]
```

### How Embedded Metadata Works

1. **Partition Fields**: For partitioned datasets, values are extracted from directory names (e.g., `accession=SRR123` → `SRR123`)
2. **Data Fields**: For single files, distinct values are extracted via HuggingFace Datasets Server API
3. **Synthetic Config**: A synthetic metadata config is created with extracted values
4. **Automatic Pairing**: The synthetic metadata automatically applies to the source config

### Metadata Extraction Priority

The system tries metadata sources in this order:
1. **Explicit metadata configs** with `applies_to` field
2. **Automatic type-based pairing** between data configs and metadata configs
3. **Embedded metadata extraction** from `metadata_fields`

## Data File Organization

### Single Files
```yaml
data_files:
- split: train
  path: single_file.parquet
```

### Multiple Files/Partitioned Data
```yaml
data_files:
- split: train
  path: data_directory/*/*.parquet  # Glob patterns supported
```

## Complete Example Structure

```yaml
license: mit
language: [en]
tags: [biology, genomics, transcription-factors]
pretty_name: "Example Genomics Dataset"
size_categories: [100K<n<1M]

configs:
- config_name: genomic_features
  description: Gene annotations and regulatory features
  dataset_type: genomic_features
  data_files:
  - split: train
    path: features.parquet
  dataset_info:
    features:
    - name: gene_id
      dtype: string
      description: Systematic gene identifier
    - name: chr
      dtype: string
      description: Chromosome name
    - name: start
      dtype: int64
      description: Gene start position

- config_name: binding_data
  description: Transcription factor binding measurements
  dataset_type: annotated_features
  default: true
  data_files:
  - split: train
    path: binding.parquet
  dataset_info:
    features:
    - name: regulator_symbol
      dtype: string
      description: Transcription factor name
      role: regulator_identifier
    - name: target_locus_tag
      dtype: string
      description: Target gene systematic identifier
      role: target_identifier
    - name: target_symbol
      dtype: string
      description: Target gene name
      role: target_identifier
    - name: binding_score
      dtype: float64
      description: Quantitative binding measurement
      role: quantitative_measure

- config_name: experiment_metadata
  description: Experimental conditions and sample information
  dataset_type: metadata
  applies_to: ["genomic_features", "binding_data"]
  data_files:
  - split: train
    path: metadata.parquet
  dataset_info:
    features:
    - name: sample_id
      dtype: string
      description: Unique sample identifier
    - name: experimental_condition
      dtype: string
      description: Experimental treatment or condition
    - name: publication_doi
      dtype: string
      description: DOI of associated publication
```

## Validation

The tfbpapi will validate:
1. Required fields are present (`dataset_type`, feature definitions)
2. Data types match the schema
3. Referenced files exist in the repository
4. Partitioning configuration is consistent with data structure

## Best Practices

1. **Descriptive names**: Use clear, descriptive field names and descriptions
2. **Consistent identifiers**: Use standard gene naming conventions
3. **Appropriate types**: Choose data types that balance precision and storage efficiency
4. **Complete metadata**: Include publication DOIs, processing methods, and data provenance
5. **Logical partitioning**: Partition large datasets by experimental conditions or samples
6. **Default config**: Mark the most commonly used configuration as `default: true`