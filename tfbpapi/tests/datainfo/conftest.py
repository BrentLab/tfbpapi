"""Shared fixtures and test data for datainfo tests."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def sample_dataset_card_data():
    """Sample dataset card data for testing."""
    return {
        "license": "mit",
        "language": ["en"],
        "tags": ["biology", "genomics", "yeast"],
        "pretty_name": "Test Genomics Dataset",
        "size_categories": ["100K<n<1M"],
        "configs": [
            {
                "config_name": "genomic_features",
                "description": "Gene annotations and regulatory features",
                "dataset_type": "genomic_features",
                "default": True,
                "data_files": [{"split": "train", "path": "features.parquet"}],
                "dataset_info": {
                    "features": [
                        {
                            "name": "gene_id",
                            "dtype": "string",
                            "description": "Systematic gene identifier",
                        },
                        {
                            "name": "gene_symbol",
                            "dtype": "string",
                            "description": "Standard gene symbol",
                        },
                        {
                            "name": "chromosome",
                            "dtype": "string",
                            "description": "Chromosome identifier",
                        },
                        {
                            "name": "start",
                            "dtype": "int64",
                            "description": "Gene start position",
                        },
                        {
                            "name": "end",
                            "dtype": "int64",
                            "description": "Gene end position",
                        },
                    ]
                },
            },
            {
                "config_name": "binding_data",
                "description": "Transcription factor binding measurements",
                "dataset_type": "annotated_features",
                "metadata_fields": ["regulator_symbol", "experimental_condition"],
                "data_files": [{"split": "train", "path": "binding/*.parquet"}],
                "dataset_info": {
                    "features": [
                        {
                            "name": "regulator_symbol",
                            "dtype": "string",
                            "description": "Transcription factor name",
                        },
                        {
                            "name": "target_gene",
                            "dtype": "string",
                            "description": "Target gene identifier",
                        },
                        {
                            "name": "experimental_condition",
                            "dtype": "string",
                            "description": "Experimental treatment condition",
                        },
                        {
                            "name": "binding_score",
                            "dtype": "float64",
                            "description": "Quantitative binding measurement",
                        },
                    ]
                },
            },
            {
                "config_name": "genome_map_data",
                "description": "Genome-wide signal tracks",
                "dataset_type": "genome_map",
                "data_files": [
                    {
                        "split": "train",
                        "path": "tracks/regulator=*/experiment=*/*.parquet",
                    }
                ],
                "dataset_info": {
                    "features": [
                        {
                            "name": "chr",
                            "dtype": "string",
                            "description": "Chromosome identifier",
                        },
                        {
                            "name": "pos",
                            "dtype": "int32",
                            "description": "Genomic position",
                        },
                        {
                            "name": "signal",
                            "dtype": "float32",
                            "description": "Signal intensity",
                        },
                    ],
                    "partitioning": {
                        "enabled": True,
                        "partition_by": ["regulator", "experiment"],
                    },
                },
            },
            {
                "config_name": "experiment_metadata",
                "description": "Experimental conditions and sample information",
                "dataset_type": "metadata",
                "applies_to": ["binding_data"],
                "data_files": [{"split": "train", "path": "metadata.parquet"}],
                "dataset_info": {
                    "features": [
                        {
                            "name": "sample_id",
                            "dtype": "string",
                            "description": "Unique sample identifier",
                        },
                        {
                            "name": "experimental_condition",
                            "dtype": "string",
                            "description": "Experimental treatment or condition",
                        },
                        {
                            "name": "publication_doi",
                            "dtype": "string",
                            "description": "DOI of associated publication",
                        },
                    ]
                },
            },
        ],
    }


@pytest.fixture
def minimal_dataset_card_data():
    """Minimal valid dataset card data."""
    return {
        "configs": [
            {
                "config_name": "test_config",
                "description": "Test configuration",
                "dataset_type": "genomic_features",
                "data_files": [{"split": "train", "path": "test.parquet"}],
                "dataset_info": {
                    "features": [
                        {
                            "name": "test_field",
                            "dtype": "string",
                            "description": "Test field",
                        }
                    ]
                },
            }
        ]
    }


@pytest.fixture
def invalid_dataset_card_data():
    """Invalid dataset card data for testing validation errors."""
    return {
        "configs": [
            {
                "config_name": "invalid_config",
                "description": "Invalid configuration",
                # Missing required dataset_type field
                "data_files": [{"split": "train", "path": "test.parquet"}],
                "dataset_info": {"features": []},  # Empty features list
            }
        ]
    }


@pytest.fixture
def sample_repo_structure():
    """Sample repository structure data."""
    return {
        "repo_id": "test/dataset",
        "files": [
            {"path": "features.parquet", "size": 2048000, "is_lfs": True},
            {"path": "binding/part1.parquet", "size": 1024000, "is_lfs": True},
            {
                "path": "tracks/regulator=TF1/experiment=exp1/data.parquet",
                "size": 5120000,
                "is_lfs": True,
            },
            {
                "path": "tracks/regulator=TF1/experiment=exp2/data.parquet",
                "size": 4096000,
                "is_lfs": True,
            },
            {
                "path": "tracks/regulator=TF2/experiment=exp1/data.parquet",
                "size": 3072000,
                "is_lfs": True,
            },
        ],
        "partitions": {"regulator": {"TF1", "TF2"}, "experiment": {"exp1", "exp2"}},
        "total_files": 5,
        "last_modified": "2023-12-01T10:30:00Z",
    }


@pytest.fixture
def sample_size_info():
    """Sample size information data."""
    return {
        "dataset": "test/dataset",
        "num_bytes": 15360000,
        "num_rows": 150000,
        "download_size": 12288000,
        "dataset_size": 15360000,
    }


@pytest.fixture
def mock_hf_card_fetcher():
    """Mock HfDataCardFetcher instance."""
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = {}
    return mock_fetcher


@pytest.fixture
def mock_hf_structure_fetcher():
    """Mock HfRepoStructureFetcher instance."""
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = {}
    mock_fetcher.get_partition_values.return_value = []
    mock_fetcher.get_dataset_files.return_value = []
    return mock_fetcher


@pytest.fixture
def mock_hf_size_fetcher():
    """Mock HfSizeInfoFetcher instance."""
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = {}
    return mock_fetcher


@pytest.fixture
def test_repo_id():
    """Standard test repository ID."""
    return "test/genomics-dataset"


@pytest.fixture
def test_token():
    """Test HuggingFace token."""
    return "test_hf_token_12345"


@pytest.fixture
def sample_feature_info():
    """Sample feature information for testing."""
    return {
        "name": "gene_symbol",
        "dtype": "string",
        "description": "Standard gene symbol (e.g., HO, GAL1)",
    }


@pytest.fixture
def sample_partitioning_info():
    """Sample partitioning information."""
    return {
        "enabled": True,
        "partition_by": ["regulator", "condition"],
        "path_template": "data/regulator={regulator}/condition={condition}/*.parquet",
    }


@pytest.fixture
def sample_data_file_info():
    """Sample data file information."""
    return {"split": "train", "path": "genomic_features.parquet"}
