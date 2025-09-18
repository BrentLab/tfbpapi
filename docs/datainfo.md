# DataInfo Package

The `datainfo` package provides dataset information management for HuggingFace datasets. It enables exploration of dataset metadata, structure, and relationships without loading actual genomic data.

## Overview

The datainfo package consists of three main components:

- **DataCard**: High-level interface for exploring dataset metadata
- **Fetchers**: Low-level components for retrieving data from HuggingFace Hub
- **Models**: Pydantic models for validation and type safety

## Main Interface

### DataCard
::: tfbpapi.datainfo.datacard.DataCard
    options:
      show_root_heading: true
      show_source: true

The `DataCard` class is the primary interface for exploring HuggingFace datasets. It provides methods to:

- Discover dataset configurations and types
- Explore feature schemas and data types
- Understand metadata relationships
- Extract field values and experimental conditions
- Navigate partitioned dataset structures

## Data Models

### Core Models
::: tfbpapi.datainfo.models.DatasetCard
    options:
      show_root_heading: true

::: tfbpapi.datainfo.models.DatasetConfig
    options:
      show_root_heading: true

::: tfbpapi.datainfo.models.FeatureInfo
    options:
      show_root_heading: true

### Dataset Types
::: tfbpapi.datainfo.models.DatasetType
    options:
      show_root_heading: true

### Relationship Models
::: tfbpapi.datainfo.models.MetadataRelationship
    options:
      show_root_heading: true

::: tfbpapi.datainfo.models.ExtractedMetadata
    options:
      show_root_heading: true

## Data Fetchers

### HuggingFace Integration
::: tfbpapi.datainfo.fetchers.HfDataCardFetcher
    options:
      show_root_heading: true

::: tfbpapi.datainfo.fetchers.HfRepoStructureFetcher
    options:
      show_root_heading: true

::: tfbpapi.datainfo.fetchers.HfSizeInfoFetcher
    options:
      show_root_heading: true

## Usage Examples

### Basic Dataset Exploration

```python
from tfbpapi.datainfo import DataCard

# Initialize DataCard for a repository
card = DataCard('BrentLab/rossi_2021')

# Get repository overview
repo_info = card.get_repository_info()
print(f"Dataset: {repo_info['pretty_name']}")
print(f"Configurations: {repo_info['num_configs']}")

# Explore configurations
for config in card.configs:
    print(f"{config.config_name}: {config.dataset_type.value}")
```

### Understanding Dataset Structure

```python
# Get detailed config information
config_info = card.explore_config('metadata')
print(f"Features: {config_info['num_features']}")

# Check for partitioned data
if 'partitioning' in config_info:
    partition_info = config_info['partitioning']
    print(f"Partitioned by: {partition_info['partition_by']}")
```

### Metadata Relationships

```python
# Discover metadata relationships
relationships = card.get_metadata_relationships()
for rel in relationships:
    print(f"{rel.data_config} -> {rel.metadata_config} ({rel.relationship_type})")
```

## Integration with HfQueryAPI

The datainfo package is designed to work seamlessly with `HfQueryAPI` for efficient data loading:

```python
from tfbpapi import HfQueryAPI
from tfbpapi.datainfo import DataCard

# Explore dataset structure first
card = DataCard('BrentLab/rossi_2021')
config_info = card.explore_config('genome_map')

# Use insights to load data efficiently
query_api = HfQueryAPI('BrentLab/rossi_2021')
data = query_api.get_pandas('genome_map',
                           filters={'run_accession': 'SRR123456'})
```

For a complete tutorial, see the [DataCard Tutorial](tutorials/datacard_tutorial.ipynb).