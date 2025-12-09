# HuggingFace Dataset Card Format

This document describes the expected YAML metadata format for HuggingFace dataset
repositories used with the tfbpapi package. The metadata is defined in the repository's
README.md file, at the top in a yaml block, and provides structured information about
the dataset configuration and contents.  

This documentation is intended for developers preparing or augmenting a huggingface
dataset repository to be compatible with tfbpapi. Before reading, please review the
[BrentLab/hackett_2020](https://huggingface.co/datasets/BrentLab/hackett_2020/blob/main/README.md) 
datacard as an example of a complete implementation of a simple repository. After
reviewing Hackett 2020 and this documentation, it might be helpful to review a more
complex example such as:

- [BrentLab/barkai_compendium](https://huggingface.co/datasets/BrentLab/barkai_compendium):
  This contains a `genome_map` partitioned dataset with separate metadata applied via
  the `applies_to` field. 
- [Brentlab/rossi_2021](https://huggingface.co/datasets/BrentLab/rossi_2021):
  This contains multiple `annotated_features` datasets with embedded metadata
- [Brentlab/yeast_genomic_features](https://huggingface.co/datasets/BrentLab/yeast_genomic_features):
  This contains a simple `genomic_features` dataset used as a reference for other
  datasets in the collection.

## Dataset Types

The `dataset_type` field is a property of each config (hierarchically under
`config_name`). `tfbpapi` recognizes the following dataset types:

### 1. `genomic_features`
Static information about genomic features (genes, promoters, etc.)
- **Use case**: Gene annotations, regulatory classifications, static feature data
- **Structure**: One row per genomic feature
- **Required fields**: Usually includes gene identifiers, coordinates, classifications

### 2. `annotated_features`
Quantitative data associated with genomic features. A field `sample_id` should exist
to identify single experiments in a single set of conditions.
- **Use case**: Expression data, binding scores, differential expression results
- **Structure**: Each sample will have one row per genomic feature measured. The
  role `quantitative_measure` should be used to identify measurement columns.
- **Common fields**: `regulator_*`, `target_*` fields with the roles
  `regulator_identifier` and `target_identifier` respectively. Fields with the role
  `quantitative_measure` for measurements.

### 3. `genome_map`
Position-level data across genomic coordinates
- **Use case**: Signal tracks, coverage data, genome-wide binding profiles
- **Structure**: Position-value pairs, often large datasets
- **Required fields**: `chr` (chromosome), `pos` (position), signal values

### 4. `metadata`
Experimental metadata and sample descriptions
- **Use case**: Sample information, experimental conditions, protocol details
- **Structure**: One row per sample
- **Common fields**: Sample identifiers, experimental conditions, publication info
- **Special field**: `applies_to` - Optional list of config names this metadata applies to

### 5. `qc_data`
Quality control metrics and assessments
- **Use case**: QC metrics derived from raw or processed data, cross-dataset quality
  assessments, validation metrics
- **Structure**: One row per sample, measurement, or QC evaluation
- **Common fields**: QC metrics, quality flags, threshold references, possibly
  references to source datasets
- **Note**: QC datasets can be derived from single or multiple source configs within
  a repository or across repositories

## Experimental Conditions

Experimental conditions can be specified in three ways:
1. **Top-level** `experimental_conditions`: Apply to all configs in the repository.
  Use when experimental parameters are common across all datasets. This will occur
  at the same level as `configs`
2. **Config-level** `experimental_conditions`: Apply to a specific config
  ([dataset](#dataset)). Use when certain datasets have experimental parameters that
  are not shared by all other datasets in the [repository](#huggingface-repo), but
  are common to all [samples](#sample) within that dataset.
3. **Field-level** with `role: experimental_condition` ([feature-roles](#feature-roles)): For
  per-sample or per-measurement variation in experimental conditions stored as
  data columns. This is specified in the
  `dataset_info.features` ([feature-definitions](#feature-definitions)) section of a config.

**Example of all three methods:**
```yaml
# Top-level experimental conditions (apply to all configs)
experimental_conditions:
  environmental_conditions:
    temperature_celsius: 30
configs:
# The overexpression_data dataset has an additional experimental condition that is
# specific to this dataset and applied to all samples (strain_background) in addition
# to a field (mechanism) that varies per sample and is identified by the
# role experimental_condition.
- config_name: overexpression_data
  description: TF overexpression perturbation data
  dataset_type: annotated_features
  experimental_conditions:
    strain_background: "BY4741"
  data_files:
    - split: train
      path: overexpression.parquet
  dataset_info:
    features:
      - name: time
        dtype: float
        description: Time point in minutes
        role: experimental_condition
      - name: mechanism
        dtype: string
        description: Induction mechanism (GEV or ZEV)
        role: experimental_condition
      - name: log2_ratio
        dtype: float
        description: Log2 fold change
        role: quantitative_measure
```

### Environmental Conditions

Environmental conditions are nested under `experimental_conditions` and describe the
physical and chemical environment in which samples were cultivated. This includes
growth media specifications, temperature, cultivation method, and other environmental
parameters.

#### Core Environmental Fields

The following fields are supported within `environmental_conditions`:

- **temperature_celsius** (float): Growth temperature in Celsius
- **cultivation_method** (string): Method of cultivation (e.g., "liquid_culture", "plate", "chemostat")
- **growth_phase_at_harvest** (object): Growth phase information (see [Growth Phase Specification](#growth-phase-specification))
- **media** (object): Growth medium specification (see [Growth Media Specification](#growth-media-specification))
- **chemical_treatment** (object): Chemical treatment information (see [Chemical Treatments](#chemical-treatments))
- **drug_treatment** (object): Drug treatment (same structure as chemical_treatment)
- **heat_treatment** (object): Heat treatment specification
- **temperature_shift** (object): Temperature shift for heat shock experiments (see [Temperature Shifts](#temperature-shifts))
- **induction** (object): Induction system for expression experiments (see [Induction Systems](#induction-systems))
- **incubation_duration_hours** (float): Total incubation duration in hours
- **incubation_duration_minutes** (int): Total incubation duration in minutes
- **description** (string): Additional descriptive information

#### Growth Phase Specification

Growth phase at harvest can be specified using:

```yaml
growth_phase_at_harvest:
  stage: mid_log_phase           # or: early_log_phase, late_log_phase, stationary_phase, etc.
  od600: 0.6                     # Optical density at 600nm
  od600_tolerance: 0.1           # Optional: measurement tolerance
  description: "Additional context"
```

**Note**: The field `phase` is accepted as an alias for `stage` for backward compatibility.

Recognized stage values:
- `mid_log_phase`, `early_log_phase`, `late_log_phase`
- `stationary_phase`, `early_stationary_phase`, `overnight_stationary_phase`
- `mid_log`, `early_log`, `late_log`, `exponential_phase`

#### Chemical Treatments

Chemical treatments (including drugs) are specified with:

```yaml
chemical_treatment:
  compound: rapamycin            # Chemical compound name
  concentration_percent: 0.001   # Concentration as percentage
  duration_minutes: 20           # Treatment duration in minutes
  duration_hours: 0.33           # Alternative: duration in hours
  target_pH: 4.0                 # Optional: target pH for pH adjustments
  description: "TOR inhibition"  # Optional: additional context
```

The `drug_treatment` field uses the same structure and is interchangeable with `chemical_treatment`.

#### Temperature Shifts

For heat shock and temperature shift experiments:

```yaml
temperature_shift:
  initial_temperature_celsius: 30
  temperature_shift_celsius: 37
  temperature_shift_duration_minutes: 45
  description: "Heat shock treatment"
```

#### Induction Systems

For expression induction systems (e.g., GAL, estradiol-inducible):

```yaml
induction:
  inducer:
    compound: D-galactose
    concentration_percent: 2
  duration_hours: 3
  duration_minutes: 180          # Alternative to duration_hours
  description: "GAL promoter induction"
```

#### Growth Media Specification

The `media` field specifies the growth medium used in an experiment. Media is nested
under `environmental_conditions` and can be specified at the top-level, config-level,
or within field-level definitions depending on whether they are common across all
datasets, specific to a config, or vary per sample.

##### Media Structure

Each media specification has the following required structure:

```yaml
experimental_conditions:
  environmental_conditions:
    media:
      name: string              # Canonical or descriptive media name (see below)
      carbon_source:            # Required
        - compound: string      # Chemical compound name
          concentration_percent: float
      nitrogen_source:          # Required
        - compound: string      # Chemical compound name
          concentration_percent: float
      phosphate_source:         # Optional
        - compound: string
          concentration_percent: float
      additives:                # Optional: for additional media components
        - compound: string      # e.g., butanol for filamentation
          concentration_percent: float
          description: string
```

Both `carbon_source` and `nitrogen_source` are **required fields**. Each can contain
one or more compound entries with their respective concentrations specified as a
percentage.

**Handling Unknown Values**: When a component is truly unknown or not reported in the
source publication, omit the field or use `null`. Do NOT use the string `"unspecified"` as
a compound name, as this will generate validation warnings.

##### Canonical Media Names

Three base media types are standardized across the collection:

1. **minimal**
   - Minimal defined medium with inorganic nitrogen source
   - Typically used for targeted nutrient deprivation studies
   - Example: Hackett 2020

2. **synthetic_complete**
   - Defined medium with amino acid supplements
   - Contains yeast nitrogen base (without amino acids) plus amino acid dropout mix
   - Used as baseline in many stress studies
   - Example: Kemmeren 2014, Mahendrawada 2025, Harbison 2004

3. **YPD** (yeast peptone dextrose)
   - Rich, complex medium with yeast extract and peptone as nitrogen sources
   - Used as standard rich-media baseline condition
   - Also known as: yeast_peptone_dextrose, yeast_extract_peptone (context-dependent)
   - Example: Hu Reimand 2010, Harbison 2004, Rossi 2021, Barkai compendium

**Descriptive Media Names**: While the canonical names above are preferred, descriptive
variations that provide additional specificity are acceptable (e.g.,
`synthetic_complete_dextrose`, `selective_medium`, `synthetic_complete_minus_uracil`).
The key requirement is that the actual media composition be fully specified in the
`carbon_source`, `nitrogen_source`, and optional `phosphate_source` and `additives` fields.

##### Specifying Carbon and Nitrogen Sources

###### Carbon Sources

Common carbon sources in yeast media:

```yaml
carbon_source:
  - compound: D-glucose
    concentration_percent: 2
```

Typical values: D-glucose, D-galactose, D-raffinose, D-dextrose

Concentrations are expressed as a percentage (e.g., 2% glucose).

###### Nitrogen Sources

Nitrogen sources vary by media type:

**For synthetic_complete and minimal media:**
```yaml
nitrogen_source:
  - compound: yeast_nitrogen_base
    concentration_g_per_l: 6.71
    specifications:
      - without_amino_acids
      - without_ammonium_sulfate
  - compound: ammonium_sulfate
    # if specified differently in the paper, add the authors'
    # specification in a comment
    concentration_percent: 0.5
  - compound: amino_acid_dropout_mix
    # lastname et al 2025 used 20 g/L
    concentration_percent: 2
```

**For YPD media:**
```yaml
nitrogen_source:
  - compound: yeast_extract
    concentration_percent: 1
  - compound: peptone
    concentration_percent: 2
```

##### Media Examples

**Minimal Medium**
```yaml
experimental_conditions:
  environmental_conditions:
    media:
      name: minimal
      carbon_source:
        - compound: D-glucose
          concentration_percent: 2
      nitrogen_source:
        - compound: ammonium_sulfate
          concentration_g_per_l: 5
```

**Synthetic Complete (Base)**
```yaml
experimental_conditions:
  environmental_conditions:
    media:
      name: synthetic_complete
      carbon_source:
        - compound: D-glucose
          concentration_percent: 2
      nitrogen_source:
        - compound: yeast_nitrogen_base
          # lastname et al 2025 used 6.71 g/L
          concentration_percent: 0.671
          specifications:
            - without_amino_acids
            - without_ammonium_sulfate
        - compound: ammonium_sulfate
          # lastname et al 2025 used 5 g/L
          concentration_percent: 0.5
        - compound: amino_acid_dropout_mix
          # lastname et al 2025 used 2 g/L
          concentration_percent: 0.2
```

**Synthetic Complete with Alternative Carbon Source**
```yaml
experimental_conditions:
  environmental_conditions:
    media:
      name: synthetic_complete
      carbon_source:
        - compound: D-galactose
          concentration_percent: 2
      nitrogen_source:
        - compound: yeast_nitrogen_base
          # lastname et al 2025 used 6.71 g/L
          concentration_percent: 0.671
          specifications:
            - without_amino_acids
            - without_ammonium_sulfate
        - compound: ammonium_sulfate
          # lastname et al 2025 used 5 g/L
          concentration_percent: 0.5
        - compound: amino_acid_dropout_mix
          # lastname et al 2025 used 2 g/L
          concentration_percent: 0.2
```

**YPD**
```yaml
experimental_conditions:
  environmental_conditions:
    media:
      name: YPD
      carbon_source:
        - compound: D-glucose
          concentration_percent: 2
      nitrogen_source:
        - compound: yeast_extract
          concentration_percent: 1
        - compound: peptone
          concentration_percent: 2
```

##### Selective/Dropout Media Variants

When a dataset uses a selective medium with specific amino acid or nutrient dropouts,
specify this using the base `synthetic_complete` name and adjust the `nitrogen_source`
to reflect the modified composition:

```yaml
experimental_conditions:
  environmental_conditions:
    media:
      name: synthetic_complete
      carbon_source:
        - compound: D-glucose
          concentration_percent: 2
      nitrogen_source:
        - compound: yeast_nitrogen_base
          # lastname et al 2025 used 6.71 g/L
          concentration_percent: 0.671
          specifications:
            - without_amino_acids
            - without_ammonium_sulfate
        - compound: ammonium_sulfate
          # lastname et al 2025 used 5 g/L
          concentration_percent: 0.5
        - compound: amino_acid_dropout_mix
          # lastname et al 2025 used 2 g/L
          concentration_percent: 0.2
          specifications:
            - minus_uracil
            - minus_histidine
            - minus_leucine
```

##### Media in Field-Level Definitions

When media varies per sample and is captured in a categorical field with definitions:

```yaml
- name: condition
  dtype:
    class_label:
      names: ["standard", "galactose"]
  role: experimental_condition
  definitions:
    standard:
      environmental_conditions:
        media:
          name: YPD
          carbon_source:
            - compound: D-glucose
              concentration_percent: 2
          nitrogen_source:
            - compound: yeast_extract
              concentration_percent: 1
            - compound: peptone
              concentration_percent: 2
    galactose:
      environmental_conditions:
        media:
          name: YPD
          carbon_source:
            - compound: D-galactose
              concentration_percent: 2
          nitrogen_source:
            - compound: yeast_extract
              concentration_percent: 1
            - compound: peptone
              concentration_percent: 2
```

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

### Categorical Fields with Value Definitions

For fields with `role: experimental_condition` that contain categorical values, you can
provide structured definitions for each value using the `definitions` field. This allows
machine-parsable specification of what each condition value means experimentally:
```yaml
- name: condition
  dtype:
    class_label:
      names: ["standard", "heat_shock"]
  role: experimental_condition
  description: Growth condition of the sample
  definitions:
    standard:
      environmental_conditions:
        media:
          name: synthetic_complete
          carbon_source:
            - compound: D-glucose
              concentration_percent: 2
          nitrogen_source:
            - compound: yeast_nitrogen_base
              concentration_g_per_l: 6.71
              specifications:
                - without_amino_acids
                - without_ammonium_sulfate
            - compound: ammonium_sulfate
              concentration_g_per_l: 5
            - compound: amino_acid_dropout_mix
              concentration_g_per_l: 2
    heat_shock:
      environmental_conditions:
        temperature_celsius: 37
        duration_minutes: 10
```

Each key in `definitions` must correspond to a possible value in the field.
The structure under each value provides experimental parameters specific to that
condition using the same nested format as `experimental_conditions` at config or
top level.

### Naming Conventions

**Gene/Feature Identifiers:**
- `(regulator/target)_locus_tag`: Systematic gene identifiers (e.g., "YJR060W"). Must
  be able to join to a genomic_features dataset. If none is specific,
  then the BrentLab/yeast_genomic_features is used
- `(regulator/target)_symbol`: Standard gene symbols (e.g., "CBF1"). Must be able to
  join to a genomic_features dataset. If none is specific,
  then the BrentLab/yeast_genomic_features is used

**Genomic Coordinates:**  
Unless otherwise noted, assume that coordinates are 0-based, half-open intervals

- `chr`: Chromosome identifier
- `start`, `end`: Genomic coordinates
- `pos`: Single position
- `strand`: Strand information (+ or -)

## Feature Roles

The optional `role` field provides semantic meaning to features, especially useful
for annotated_features datasets. The following roles are recognized by tfbpapi:

- **target_identifier**: Identifies target genes/features (e.g., `target_locus_tag`,
  `target_symbol`)
- **regulator_identifier**: Identifies regulatory factors (e.g., `regulator_locus_tag`,
  `regulator_symbol`)
- **quantitative_measure**: Quantitative measurements (e.g., `binding_score`,
  `expression_level`, `p_value`)
- **experimental_condition**: Experimental conditions or metadata
  (can include `definitions` field for categorical values)
- **genomic_coordinate**: Positional information (`chr`, `start`, `end`, `pos`)

**Validation**: Only these specific role values are accepted. Other values (e.g., `"identifier"`)
will cause validation errors.

## Strain Background and Definitions

The `strain_background` field can appear in two locations:

1. **Top-level or config-level** `experimental_conditions`:
   ```yaml
   experimental_conditions:
     strain_background:
       genotype: BY4741
       mating_type: MATa
       markers: [his3Δ1, leu2Δ0, met15Δ0, ura3Δ0]
   ```

2. **Within field-level definitions** for condition-specific strain information:
   ```yaml
   - name: heat_shock
     dtype:
       class_label:
         names: ["control", "treated"]
     role: experimental_condition
     definitions:
       treated:
         environmental_conditions:
           temperature_celsius: 37
         strain_background:
           genotype: W303_derivative
           description: "Heat-sensitive strain"
   ```

The `strain_background` field accepts flexible structure as a dictionary to accommodate
varying levels of detail about strain information.


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

For metadata configs, you can explicitly specify which other configs the metadata
applies to using the `applies_to` field. This provides more control than automatic
type-based matching.

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

When no explicit metadata config exists, you can extract metadata directly from the
dataset's own files using the `metadata_fields` field. This specifies which fields
should be treated as metadata.

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

1. **Partition Fields**: For partitioned datasets, values are extracted from directory
  names (e.g., `accession=SRR123` to `SRR123`)
2. **Data Fields**: For single files, distinct values are extracted via HuggingFace
  Datasets Server API
3. **Synthetic Config**: A synthetic metadata config is created with extracted values
4. **Automatic Pairing**: The synthetic metadata automatically applies to the source
  config

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

## Terms and definitions

### field/feature/attribute/column
In a collection of samples (see below), the fields record information about the
record. For example, if there are two samples each of which report results for 6000
genes and the way in which the samples differ is by growth media, then growth_media
would be a feature with two levels, eg YPD and SC. If the two samples are stored in
the same parquet file, then there would be a column where the entry for all 6000
rows of the first sample would be YPD and the entry for all 6000 rows of the second
sample would be SC.

### record/row
A row in a table, or a single observation in a single sample (see below).

### metadata
Data about data. However, there are multiple objects to which metadata is attached in
our usage, in particular at the dataset level and at the repo level (see below for
those terms).

### sample
The result of a single biological experiment. For example, if a given dataset has 20
regulators, in 3 replicates in 2 conditions, then there would be 20×3×2 samples.
If the way the results are reported is over 6000 genes, then we would expect all
20×3×2 of those samples to have 6000 records.

### huggingface repo
HuggingFace is a thin layer on top of GitHub. HuggingFace repos are GitHub repos with
additional functionality.

### datacard
A README file in the HuggingFace repo. In HuggingFace, this is called a datacard and
has an additional YAML section at the top. This YAML section stores information on
the repo and is extensible. It is in this YAML section that we record a defined set
of attributes and features that allow us to search/filter/subset the data in the
collection (see below). See the datacard format documentation for a full description.

### dataset
In our HuggingFace repos, we store one or more datasets. These datasets have
defined types. In general, we try to refer to datasets by the first author and year
of the paper from which they originate, eg 'Mahendrawada 2025'. However, the
distinction between a dataset and a repo can be complicated, as in the case of
Mahendrawada 2025 there is ChEC-seq, ChIP-seq and RNA-seq data. Each of those may be
provided in multiple datasets, eg one which was reported by the authors, and another
reprocessed in our lab. A dataset should refer to a single one of those collections
and may require further specification beyond the first author's name and year published.

### huggingface collection
HuggingFace allows you to group repositories together, which is what we are doing
with all repos storing data related to the yeast database project.

### regulator
A superset that includes "TF" or "transcription factor". These are proteins which
are assayed for their effect on gene expression.

### target
Genes on which the regulator's effect is measured.

### tfbpapi
A Python package which provides the interface to the HuggingFace collection.

### active set (of samples)
In order to conduct analysis, a user will need to define a set of samples. A sample
(see definition above) is defined by the metadata features, eg regulator_locus_tag.
If the user is interested in all datasets in which this regulator exists, then the
active set would be the set of samples, across the entire collection (see HuggingFace
collection above), with this regulator_locus_tag. The user may choose to filter on
additional features in order to further refine the active set (eg, if a different
dataset has 2 conditions for that regulator, then the user may wish to only retain
1 of those conditions in their active set. They may wish to completely exclude a
different dataset, etc).