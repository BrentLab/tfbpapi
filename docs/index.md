# tfbpapi Documentation

## Development Commands

### Testing
- Run tests: `poetry run pytest`
- Run specific test: `poetry run pytest tfbpapi/tests/test_[module_name].py`
- Run tests with coverage: `poetry run pytest --cov=tfbpapi`

### Linting and Formatting
- Run all pre-commit checks: `poetry run pre-commit run --all-files`
- Format code with Black: `poetry run black tfbpapi/`
- Sort imports with isort: `poetry run isort tfbpapi/`
- Type check with mypy: `poetry run mypy tfbpapi/`
- Lint with flake8: `poetry run flake8 tfbpapi/`

### Installation
- Install dependencies: `poetry install`
- Install pre-commit hooks: `poetry run pre-commit install`

## Architecture

This is a Python package for interfacing with a collection of datasets hosted on Hugging Face. The modern architecture provides efficient querying, caching, and metadata management for genomic and transcriptomic datasets.

### Core Components

- **HfQueryAPI** (`tfbpapi/HfQueryAPI.py`): Main interface for querying HF datasets with intelligent downloading and SQL querying capabilities. Supports automatic dataset size detection, selective downloading, and DuckDB-based querying.

- **HfCacheManager** (`tfbpapi/HfCacheManager.py`): Manages HF cache with cleanup and size management features. Provides automatic cache cleanup based on age and size thresholds.

- **HfRankResponse** (`tfbpapi/HfRankResponse.py`): Response handling for HF-based ranking operations. Computes and analyzes "rank response" - the cumulative number of responsive targets binned by binding rank scores.

- **IncrementalAnalysisDB** (`tfbpapi/IncrementalAnalysisDB.py`): Database management for incremental analysis workflows with shared result storage.

### Dataset Information Management

- **datainfo package** (`tfbpapi/datainfo/`): Comprehensive dataset exploration and metadata management for HuggingFace datasets. Provides the `DataCard` class for exploring dataset structure, configurations, and relationships without loading actual data. Includes Pydantic models for validation and fetchers for HuggingFace Hub integration.

### Data Types

The datasets in this collection store the following types of genomic data:

- **genomic_features**: Labels and information about genomic features (e.g., parsed GTF/GFF files)
- **annotated_features**: Data quantified to features, typically genes
- **genome_map**: Data mapped to genome coordinates
- **metadata**: Additional sample information (cell types, experimental conditions, etc.)

Data is stored in Apache Parquet format, either as single files or parquet datasets (directories of parquet files).

### Error Handling

- **errors.py** (`tfbpapi/errors.py`): Custom exception classes for dataset management including `DatasetError`, `RepoTooLargeError`, `DataCardParsingError`, `HfDataFetchError`, and more.

## Configuration

- Uses Poetry for dependency management
- Python 3.11+ required
- Black formatter with 88-character line length
- Pre-commit hooks include Black, isort, flake8, mypy, and various file checks
- pytest with comprehensive testing support
- Environment variables: `HF_TOKEN`, `HF_CACHE_DIR`

## Testing Patterns

- Tests use pytest with modern testing practices
- Integration tests for HuggingFace dataset functionality
- Test fixtures for dataset operations
- Comprehensive error handling testing

### mkdocs

#### Commands

After building the environment with poetry, you can use `poetry run` or a poetry shell
to execute the following:

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

#### Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

To update the gh-pages documentation, use `poetry run mkdocs gh-deply`

