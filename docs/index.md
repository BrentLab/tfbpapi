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

This is a Python package for interacting with django.tfbindingandmodeling.com APIs. The codebase follows an object-oriented architecture with abstract base classes providing common functionality.

### Core Architecture Classes

- **AbstractAPI** (`tfbpapi/AbstractAPI.py`): Base class for all API clients with token authentication, caching (via `Cache` class), parameter validation (via `ParamsDict` class), and abstract CRUD methods.

- **AbstractRecordsAndFilesAPI** (`tfbpapi/AbstractRecordsAndFilesAPI.py`): Extends `AbstractAPI` for endpoints that serve both record metadata and associated data files. Handles tarball extraction, CSV parsing, and file caching.

- **AbstractRecordsOnlyAPI** (`tfbpapi/AbstractRecordsOnlyAPI.py`): Extends `AbstractAPI` for endpoints that only serve record metadata without file storage.

- **AbstractHfAPI** (`tfbpapi/AbstractHfAPI.py`): Abstract base for Hugging Face Hub integration, providing repository management functionality.

### Concrete API Classes

All concrete API classes inherit from either `AbstractRecordsAndFilesAPI` or `AbstractRecordsOnlyAPI`:

- `BindingAPI`, `ExpressionAPI`, `PromoterSetAPI` - Record and file APIs
- `DataSourceAPI`, `RegulatorAPI`, `GenomicFeatureAPI` - Records only APIs

### Utility Classes

- **Cache** (`tfbpapi/Cache.py`): TTL-based caching for API responses
- **ParamsDict** (`tfbpapi/ParamsDict.py`): Parameter validation against allowed keys
- **HfCacheManager** (`tfbpapi/HfCacheManager.py`): Hugging Face cache cleanup utilities

### Data Processing Utilities

- **metric_arrays** (`tfbpapi/metric_arrays.py`): Array-based metric calculations
- **rank_transforms** (`tfbpapi/rank_transforms.py`): Statistical rank transformation functions

## Configuration

- Uses Poetry for dependency management
- Python 3.11 required
- Black formatter with 88-character line length
- Pre-commit hooks include Black, isort, flake8, mypy, and various file checks
- pytest with snapshot testing support
- Environment variables: `BASE_URL`, `TOKEN`, `HF_TOKEN`, `HF_CACHE_DIR`

## Testing Patterns

- Tests use pytest with async support (`pytest-asyncio`)
- Snapshot testing with `pytest-snapshot` for API response validation
- Test fixtures in `tfbpapi/tests/conftest.py`
- Mock HTTP responses using `aioresponses` and `responses` libraries

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

